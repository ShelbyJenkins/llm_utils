use super::*;

#[derive(Clone, Debug, PartialEq)]
pub enum PromptMessageType {
    System,
    User,
    Assistant,
}

impl PromptMessageType {
    pub fn as_str(&self) -> &str {
        match self {
            PromptMessageType::System => "system",
            PromptMessageType::User => "user",
            PromptMessageType::Assistant => "assistant",
        }
    }
}

#[derive(Clone, Debug)]
pub struct PromptMessage {
    content: Option<Vec<String>>,
    pub built_prompt_hashmap: HashMap<String, String>,
    pub built_prompt_string: Option<String>,
    pub message_type: PromptMessageType,
    pub concatenator: PromptConcatenator,
}

impl PromptMessage {
    pub fn new(message_type: &PromptMessageType, concatenator: &PromptConcatenator) -> Self {
        Self {
            content: None,
            built_prompt_hashmap: HashMap::new(),
            built_prompt_string: None,
            message_type: message_type.clone(),
            concatenator: concatenator.clone(),
        }
    }

    pub fn set_content<T: AsRef<str>>(&mut self, content: T) -> &mut Self {
        if content.as_ref().is_empty() {
            panic!("PromptMessage content cannot be empty");
        }
        if let Some(existing) = &self.content {
            if existing.len() == 1 && existing.iter().any(|c| c == content.as_ref()) {
                return self;
            }
            self.built_prompt_hashmap.clear();
        }
        self.content = Some(vec![content.as_ref().to_owned()]);
        self
    }

    pub fn set_content_from_path(&mut self, content_path: &PathBuf) -> &mut Self {
        self.set_content(load_content_path(content_path))
    }

    pub fn prepend_content<T: AsRef<str>>(&mut self, content: T) -> &mut Self {
        if content.as_ref().is_empty() {
            panic!("PromptMessage content cannot be empty");
        }
        if let Some(existing) = &mut self.content {
            if existing.is_empty() || existing.iter().all(|c| c != content.as_ref()) {
                self.built_prompt_hashmap.clear();
                existing.insert(0, content.as_ref().to_owned());
            }
        } else {
            self.content = Some(vec![content.as_ref().to_owned()]);
        }
        self
    }

    pub fn prepend_content_from_path(&mut self, content_path: &PathBuf) -> &mut Self {
        self.prepend_content(load_content_path(content_path))
    }

    pub fn append_content<T: AsRef<str>>(&mut self, content: T) -> &mut Self {
        if content.as_ref().is_empty() {
            panic!("PromptMessage content cannot be empty");
        }
        if let Some(existing) = &mut self.content {
            if existing.is_empty() || existing.iter().all(|c| c != content.as_ref()) {
                self.built_prompt_hashmap.clear();
                existing.push(content.as_ref().to_owned());
            }
        } else {
            self.content = Some(vec![content.as_ref().to_owned()]);
        }
        self
    }

    pub fn append_content_from_path(&mut self, content_path: &PathBuf) -> &mut Self {
        self.append_content(load_content_path(content_path))
    }

    pub fn requires_build(&self) -> bool {
        self.content.is_some() && self.built_prompt_hashmap.is_empty()
    }

    pub fn build(&mut self) {
        if let Some(built_prompt_string) = self.build_prompt_string() {
            self.built_prompt_hashmap = HashMap::from([
                ("role".to_string(), self.message_type.as_str().to_owned()),
                ("content".to_string(), built_prompt_string.to_owned()),
            ]);
            self.built_prompt_string = Some(built_prompt_string);
        }
    }

    fn build_prompt_string(&self) -> Option<String> {
        let content = if let Some(content) = &self.content {
            if content.is_empty() {
                return None;
            } else {
                content
            }
        } else {
            return None;
        };
        let mut built_prompt_string = String::new();

        for c in content {
            if c.as_str().is_empty() {
                continue;
            }
            if !built_prompt_string.is_empty() {
                built_prompt_string.push_str(self.concatenator.as_str());
            }
            built_prompt_string.push_str(c.as_str());
        }
        if built_prompt_string.is_empty() {
            return None;
        }
        Some(built_prompt_string)
    }
}

pub fn add_system_message(messages: &mut Vec<PromptMessage>, concatenator: &PromptConcatenator) {
    if !messages.is_empty() {
        panic!("System message must be first message.");
    };
    let message = PromptMessage::new(&PromptMessageType::System, concatenator);
    messages.push(message);
}

pub fn add_user_message(messages: &mut Vec<PromptMessage>, concatenator: &PromptConcatenator) {
    if !messages.is_empty() && messages.last().unwrap().message_type == PromptMessageType::User {
        panic!("Cannot add user message when previous message is user message.");
    }
    let message = PromptMessage::new(&PromptMessageType::User, concatenator);
    messages.push(message);
}

pub fn add_assistant_message(messages: &mut Vec<PromptMessage>, concatenator: &PromptConcatenator) {
    if messages.is_empty() {
        panic!("Cannot add assistant message as first message.");
    } else if messages.last().unwrap().message_type == PromptMessageType::Assistant {
        panic!("Cannot add assistant message when previous message is assistant message.");
    };
    let message = PromptMessage::new(&PromptMessageType::Assistant, concatenator);
    messages.push(message);
}

pub fn build_messages(messages: &mut [PromptMessage]) -> Vec<HashMap<String, String>> {
    let mut prompt_messages: Vec<HashMap<String, String>> = Vec::new();
    let mut last_message_type = None;
    for (i, message) in messages.iter_mut().enumerate() {
        let message_type = &message.message_type;
        // Rule 1: System message can only be the first message
        if *message_type == PromptMessageType::System && i != 0 {
            panic!("System message can only be the first message.");
        }
        // Rule 2: First message must be either System or User
        if i == 0
            && *message_type != PromptMessageType::System
            && *message_type != PromptMessageType::User
        {
            panic!("Conversation must start with either a System or User message.");
        }
        // Rule 3: Ensure alternating User/Assistant messages after the first message
        if i > 0 {
            match (last_message_type, message_type) {
                (Some(PromptMessageType::User), PromptMessageType::Assistant) => {},
                (Some(PromptMessageType::Assistant), PromptMessageType::User) => {},
                (Some(PromptMessageType::System), PromptMessageType::User) => {},
                _ => panic!("Messages must alternate between User and Assistant after the first message (which can be System)."),
            }
        }
        last_message_type = Some(message_type.clone());
        if message.requires_build() {
            message.build();
        }
        if message.built_prompt_hashmap.is_empty() {
            eprintln!("message.built_content is empty and skipped");
            continue;
        }
        prompt_messages.push(message.built_prompt_hashmap.clone());
    }
    prompt_messages
}

impl std::fmt::Display for PromptMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message_type = match self.message_type {
            PromptMessageType::System => "System",
            PromptMessageType::User => "User",
            PromptMessageType::Assistant => "Assistant",
        };
        let message = if let Some(built_prompt_string) = &self.built_prompt_string {
            if built_prompt_string.len() > 300 {
                format!(
                    "{}...",
                    built_prompt_string.chars().take(300).collect::<String>()
                )
            } else {
                built_prompt_string.clone()
            }
        } else {
            "debug message: empty or unbuilt".to_string()
        };
        writeln!(f, "\x1b[1m{message_type}\x1b[0m:\n{:?}", message)
    }
}
