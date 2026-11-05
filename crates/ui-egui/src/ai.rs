//! The AI Assistant: an optional chat panel for the open PDF.
//!
//! **On by default in this build**: View ▸ AI Assistant (Ctrl/⌘+Shift+A) opens the panel without
//! any setup, and Preferences ▸ AI Assistant turns it off, or sets the service it talks to.
//!
//! Nothing leaves the machine until the person presses Send. The only document text that goes
//! with a question is what the panel's *Send with the question* control names — nothing, the page
//! shown, or the document — and it is read on the worker thread, never on the interface thread.
//!
//! The service is any OpenAI-compatible `POST {base_url}/chat/completions` endpoint: the address
//! is a preference, so a local server (Ollama, llama.cpp), a gateway or a hosted API all work
//! without a build of their own.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc,
};

use egui::RichText;
use serde::{Deserialize, Serialize};

use crate::{PdfCraftApp, theme};

/// The service address a fresh install offers, used only once the assistant is turned on.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
/// The model a fresh install offers.
pub const DEFAULT_MODEL: &str = "gpt-4o-mini";
/// The instructions sent before the conversation; the document text follows them.
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are a PDF assistant inside PdfCraft, a PDF reader. Answer in the \
language of the question. Be concise and quote the document where it matters. If the document text \
does not answer the question, say so instead of guessing.";
/// The most pages read for a whole-document question.
pub(crate) const MAX_CONTEXT_PAGES: usize = 20;
/// Document text sent with a question is cut here.
const MAX_CONTEXT_CHARS: usize = 40_000;
/// A reply longer than this is cut before it reaches the interface.
const MAX_REPLY_CHARS: usize = 40_000;
/// The most recent turns sent back with a question.
const MAX_HISTORY_TURNS: usize = 12;

/// The assistant's settings, kept in PdfCraft's own settings file on this computer.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// On for a fresh install (the fork's choice); Preferences ▸ AI Assistant turns it off, and
    /// off there it stays until turned back on.
    pub enabled: bool,
    /// An OpenAI-compatible service's base address (`…/v1`); `/chat/completions` is added.
    pub base_url: String,
    /// The model name the service knows.
    pub model: String,
    /// The service's key. Empty for a local server that needs none.
    pub api_key: String,
    /// Whether a question may carry document text at all (the panel can still send none).
    pub include_document_text: bool,
    /// What the model is told before the conversation.
    pub system_prompt: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            // On: the assistant is part of this build. Nothing is sent until Send is pressed,
            // and Preferences turns the whole panel off again.
            enabled: true,
            base_url: DEFAULT_BASE_URL.to_string(),
            model: DEFAULT_MODEL.to_string(),
            api_key: String::new(),
            include_document_text: true,
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
        }
    }
}

impl Settings {
    /// The settings as saved (one object in PdfCraft's settings file).
    pub(crate) fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }

    /// Settings restored from the settings file, which is untrusted: anything unreadable keeps
    /// the defaults, and nothing that is read is trusted further than the panel can use.
    pub(crate) fn from_json(v: &serde_json::Value) -> Settings {
        let mut settings = serde_json::from_value::<Settings>(v.clone()).unwrap_or_default();
        settings.sanitized()
    }

    /// Trimmed addresses and names, and the key cut to a length a header can carry.
    fn sanitized(mut self) -> Settings {
        self.base_url = self.base_url.trim().chars().take(300).collect();
        self.model = self.model.trim().chars().take(120).collect();
        self.api_key = self.api_key.trim().chars().take(400).collect();
        if self.base_url.is_empty() {
            self.base_url = DEFAULT_BASE_URL.to_string();
        }
        if self.model.is_empty() {
            self.model = DEFAULT_MODEL.to_string();
        }
        if self.system_prompt.trim().is_empty() {
            self.system_prompt = DEFAULT_SYSTEM_PROMPT.to_string();
        }
        self.system_prompt = self.system_prompt.chars().take(4_000).collect();
        self
    }
}

/// What the panel sends with a question.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Scope {
    /// No document text: a plain chat.
    Off,
    /// The page on screen.
    #[default]
    Page,
    /// The document's first pages (see [`MAX_CONTEXT_PAGES`]).
    Document,
}

/// Who said a message in the transcript.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    /// The request failed; `text` says why.
    Error,
}

impl Role {
    /// The `role` an OpenAI-compatible service expects.
    fn wire(self) -> &'static str {
        match self {
            Role::User => "user",
            // A failure is shown to the person, never sent back as the model's own words.
            Role::Assistant | Role::Error => "assistant",
        }
    }
}

/// One turn of the conversation.
#[derive(Clone, Debug)]
pub struct Message {
    pub role: Role,
    pub text: String,
}

/// What a request reads from the open document, ready to move to the worker thread.
pub(crate) struct DocContext {
    name: String,
    source: pdfcraft_engine::export::ExportSource,
    /// The pages to read (0-based).
    pages: Vec<usize>,
}

/// A request running on its own thread.
struct Pending {
    receiver: mpsc::Receiver<Result<String, String>>,
    cancel: Arc<AtomicBool>,
}

/// The AI Assistant's state: its settings, the conversation, and the request in flight.
pub struct AiAssistant {
    pub settings: Settings,
    /// The panel is showing.
    pub open: bool,
    pub messages: Vec<Message>,
    /// The question being typed.
    pub input: String,
    /// What the next question sends with it.
    pub scope: Scope,
    pending: Option<Pending>,
}

impl Default for AiAssistant {
    fn default() -> Self {
        Self { settings: Settings::default(), open: false, messages: Vec::new(), input: String::new(), scope: Scope::default(), pending: None }
    }
}

impl AiAssistant {
    /// Whether a question is waiting for its answer.
    pub(crate) fn busy(&self) -> bool {
        self.pending.is_some()
    }

    /// Take in an answer (or a failure) the worker finished, if any.
    fn poll(&mut self) {
        let received = match &self.pending {
            Some(job) => match job.receiver.try_recv() {
                Ok(result) => Some(Some(result)),
                Err(mpsc::TryRecvError::Empty) => None,
                // The worker went away without an answer: say so rather than wait for ever.
                Err(mpsc::TryRecvError::Disconnected) => Some(None),
            },
            None => None,
        };
        let Some(received) = received else { return };
        self.pending = None;
        match received {
            Some(Ok(text)) => self.messages.push(Message { role: Role::Assistant, text }),
            Some(Err(error)) => self.messages.push(Message { role: Role::Error, text: error }),
            None => {
                self.messages.push(Message { role: Role::Error, text: tl!("The request stopped unexpectedly.").to_string() });
            }
        }
    }

    /// Stop waiting for the answer in flight, if any.
    fn cancel(&mut self) {
        if let Some(job) = self.pending.take() {
            job.cancel.store(true, Ordering::Relaxed);
        }
    }

    /// Forget the conversation.
    fn clear(&mut self) {
        self.cancel();
        self.messages.clear();
        self.input.clear();
    }

    /// Ask the service `system` + `history`, with `context` read on the worker thread first.
    fn start(&mut self, ctx: &egui::Context, system: String, history: Vec<Message>, context: Option<DocContext>) {
        if self.pending.is_some() {
            return;
        }
        let settings = self.settings.clone();
        let (sender, receiver) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        #[cfg(not(target_arch = "wasm32"))]
        {
            let stop = cancel.clone();
            let ctx = ctx.clone();
            let spawned = std::thread::Builder::new().name("ai-assistant".into()).spawn(move || {
                let result = run(&settings, &system, &history, context, &stop);
                let _ = sender.send(result);
                ctx.request_repaint();
            });
            match spawned {
                Ok(_) => self.pending = Some(Pending { receiver, cancel }),
                Err(error) => {
                    let text = format!("{} {error}", tl!("Couldn't start the request."));
                    self.messages.push(Message { role: Role::Error, text });
                }
            }
            return;
        }
        // A browser has no thread to wait on and no socket of its own: the panel says so.
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (ctx, settings, system, history, context, sender, receiver, cancel);
            self.messages.push(Message {
                role: Role::Error,
                text: tl!("The AI Assistant needs the desktop app; the web build has no network service to reach.").to_string(),
            });
        }
    }
}

/// Cut `text` to at most `max` characters, on a character boundary.
fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max).collect();
    out.push('…');
    out
}

/// The service's answer to one question, with the document text read first.
#[cfg(not(target_arch = "wasm32"))]
fn run(settings: &Settings, system: &str, history: &[Message], context: Option<DocContext>, cancel: &Arc<AtomicBool>) -> Result<String, String> {
    let mut system = system.to_string();
    if let Some(context) = context {
        if cancel.load(Ordering::Relaxed) {
            return Err(tl!("Stopped.").to_string());
        }
        let text = pdfcraft_engine::export::Exporter::from_source(context.source).text_of(&context.pages).unwrap_or_default();
        let text = clip(&text, MAX_CONTEXT_CHARS);
        if !text.trim().is_empty() {
            system.push_str(&format!(
                "\n\nThe user is reading \"{}\"; the text of {} page(s) of it follows between the markers. Use it as the source of truth.\n<<<PDF TEXT\n{}\nPDF TEXT>>>",
                context.name,
                context.pages.len(),
                text
            ));
        }
    }
    request(settings, &system, history, cancel)
}

/// One `chat/completions` call, read to a string.
#[cfg(not(target_arch = "wasm32"))]
fn request(settings: &Settings, system: &str, history: &[Message], cancel: &Arc<AtomicBool>) -> Result<String, String> {
    let base = settings.base_url.trim().trim_end_matches('/');
    if base.is_empty() {
        return Err(tl!("Set the service address in Preferences ▸ AI Assistant.").to_string());
    }
    let model = settings.model.trim();
    if model.is_empty() {
        return Err(tl!("Set a model name in Preferences ▸ AI Assistant.").to_string());
    }
    let mut messages: Vec<serde_json::Value> = Vec::with_capacity(history.len() + 1);
    messages.push(serde_json::json!({ "role": "system", "content": system }));
    let first = history.len().saturating_sub(MAX_HISTORY_TURNS);
    for message in &history[first..] {
        messages.push(serde_json::json!({ "role": message.role.wire(), "content": message.text }));
    }
    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": false,
    })
    .to_string();

    let agent: ureq::Agent = ureq::Agent::config_builder()
        // A refused request's own message is more useful than "status 401", so read the body.
        .http_status_as_error(false)
        .timeout_global(Some(std::time::Duration::from_secs(180)))
        .build()
        .into();
    if cancel.load(Ordering::Relaxed) {
        return Err(tl!("Stopped.").to_string());
    }
    let url = format!("{base}/chat/completions");
    let mut call = agent.post(url.as_str()).header("content-type", "application/json");
    if !settings.api_key.trim().is_empty() {
        call = call.header("authorization", format!("Bearer {}", settings.api_key.trim()));
    }
    let mut response = call
        .send(body.as_str())
        .map_err(|error| format!("{} {error}", tl!("Couldn't reach the AI service:")))?;
    let status = response.status().as_u16();
    let text = response
        .body_mut()
        .read_to_string()
        .map_err(|error| format!("{} {error}", tl!("Couldn't read the answer:")))?;

    let parsed: serde_json::Value = serde_json::from_str(&text).unwrap_or(serde_json::Value::Null);
    if !(200..300).contains(&status) {
        let detail = parsed
            .get("error")
            .and_then(|error| error.get("message"))
            .and_then(serde_json::Value::as_str)
            .or_else(|| parsed.get("message").and_then(serde_json::Value::as_str));
        let body = clip(text.trim(), 300);
        let message = match detail {
            Some(detail) => detail.to_string(),
            None if body.is_empty() => format!("{} ({status})", tl!("The AI service refused the request")),
            None => format!("{} ({status}): {body}", tl!("The AI service refused the request")),
        };
        return Err(message);
    }
    let content = parsed
        .get("choices")
        .and_then(|choices| choices.get(0))
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .trim();
    if content.is_empty() {
        let body = clip(text.trim(), 300);
        return Err(format!("{} {body}", tl!("The AI service answered with no text.")));
    }
    Ok(clip(content, MAX_REPLY_CHARS))
}

impl PdfCraftApp {
    /// View ▸ AI Assistant, the ⌘K palette and the shortcut all come here: the panel opens, or
    /// the preferences that turn it on do when it is still off.
    pub fn toggle_ai_assistant(&mut self) {
        if !self.ai.settings.enabled {
            self.notify_tr("The AI Assistant is off. Turn it on in Preferences ▸ AI Assistant, then try again.");
            self.dialog = Some(crate::Dialog::Preferences);
            return;
        }
        if self.ai.open {
            self.ai.open = false;
            self.ai.cancel();
        } else {
            self.ai.open = true;
        }
    }

    /// Send the typed question, with the document text the panel names.
    pub(crate) fn ai_send(&mut self, ctx: &egui::Context) {
        if self.ai.busy() {
            return;
        }
        let prompt = self.ai.input.trim().to_string();
        if prompt.is_empty() {
            return;
        }
        self.ai.input.clear();
        self.ai.messages.push(Message { role: Role::User, text: prompt });
        let context = self.ai_context();
        let history: Vec<Message> = self.ai.messages.iter().filter(|m| m.role != Role::Error).cloned().collect();
        let system = self.ai.settings.system_prompt.clone();
        self.ai.start(ctx, system, history, context);
    }

    /// The document text the next question carries, as plain values for the worker thread.
    fn ai_context(&self) -> Option<DocContext> {
        if !self.ai.settings.include_document_text || self.ai.scope == Scope::Off {
            return None;
        }
        let view = self.views.get(self.active?)?;
        let doc = self.session.get(view.id)?;
        let pages: Vec<usize> = match self.ai.scope {
            Scope::Page => vec![view.current],
            Scope::Document => (0..doc.info.pages.len()).take(MAX_CONTEXT_PAGES).collect(),
            Scope::Off => return None,
        };
        if pages.is_empty() {
            return None;
        }
        Some(DocContext { name: doc.name.clone(), source: doc.export_source(), pages })
    }
}

/// Draw the panel, when it is on and open, above the document.
pub(crate) fn show(app: &mut PdfCraftApp, ctx: &egui::Context) {
    if !app.ai.settings.enabled {
        // Turned off while a request was in flight: stop it and put the panel away.
        app.ai.cancel();
        app.ai.open = false;
        return;
    }
    app.ai.poll();
    if !app.ai.open {
        return;
    }
    let t = theme::Tokens::get(ctx);
    let mut open = true;
    let mut send = false;
    let mut stop = false;
    let mut clear = false;
    {
        let ai = &mut app.ai;
        egui::Window::new(tl!("AI Assistant"))
            .id(egui::Id::unique("ai-assistant-panel"))
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_size(egui::vec2(430.0, 560.0))
            .default_pos(egui::pos2(120.0, 110.0))
            .show(ctx, |ui| {
                ui.set_min_width(380.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new(tl!("Send with the question")).font(theme::regular(12.0)).color(t.text_muted));
                    ui.selectable_value(&mut ai.scope, Scope::Off, tl!("Nothing"));
                    ui.selectable_value(&mut ai.scope, Scope::Page, tl!("This page"));
                    ui.selectable_value(&mut ai.scope, Scope::Document, tl!("The document"));
                });
                if !ai.settings.include_document_text {
                    ui.label(
                        RichText::new(tl!("Document text is turned off in Preferences ▸ AI Assistant, so only the question is sent."))
                            .font(theme::regular(11.5))
                            .color(t.text_faint),
                    );
                }
                ui.separator();
                let height = (ui.available_height() - 150.0).max(120.0);
                egui::ScrollArea::vertical().max_height(height).stick_to_bottom(true).show(ui, |ui| {
                    if ai.messages.is_empty() {
                        ui.label(
                            RichText::new(tl!(
                                "Ask about the open PDF, or anything else. Nothing leaves this computer until you press Send."
                            ))
                            .font(theme::regular(12.5))
                            .color(t.text_muted),
                        );
                    }
                    for (index, message) in ai.messages.iter().enumerate() {
                        if index > 0 {
                            ui.add_space(10.0);
                        }
                        let (title, color) = match message.role {
                            Role::User => (tl!("You"), t.text_muted),
                            Role::Assistant => (tl!("Assistant"), t.accent),
                            Role::Error => (tl!("Couldn't ask"), ui.visuals().error_fg_color),
                        };
                        ui.label(RichText::new(title).font(theme::medium(11.5)).color(color));
                        ui.add(
                            egui::Label::new(RichText::new(message.text.as_str()).font(theme::regular(13.0)).color(t.text))
                                .wrap()
                                .selectable(true),
                        );
                    }
                    if ai.busy() {
                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(RichText::new(tl!("Asking…")).font(theme::regular(12.0)).color(t.text_muted));
                        });
                    }
                });
                ui.separator();
                let edit = ui.add(
                    egui::TextEdit::multiline(&mut ai.input)
                        .hint_text(tl!("Ask a question…"))
                        .desired_rows(3)
                        .desired_width(f32::INFINITY)
                        .id_salt("ai-assistant-input"),
                );
                // ⌘/Ctrl+Enter sends without leaving the field.
                if edit.has_focus() && ui.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Enter)) {
                    send = true;
                }
                ui.horizontal(|ui| {
                    let ready = !ai.busy() && !ai.input.trim().is_empty();
                    if ui.add_enabled(ready, egui::Button::new(tl!("Send"))).clicked() {
                        send = true;
                    }
                    if ai.busy() && ui.button(tl!("Stop")).clicked() {
                        stop = true;
                    }
                    if !ai.messages.is_empty() && ui.button(tl!("Clear")).clicked() {
                        clear = true;
                    }
                });
            });
    }
    app.ai.open = open;
    if stop {
        app.ai.cancel();
    }
    if clear {
        app.ai.clear();
    }
    if send {
        app.ai_send(ctx);
    }
}
