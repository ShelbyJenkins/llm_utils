#[derive(Clone, PartialEq, Debug, Default)]
pub enum PromptConcatenator {
    DoubleNewline,
    #[default]
    SingleNewline,
    Space,
    Comma,
    Custom(String),
}

impl PromptConcatenator {
    pub fn as_str(&self) -> &str {
        match self {
            PromptConcatenator::DoubleNewline => "\n\n",
            PromptConcatenator::SingleNewline => "\n",
            PromptConcatenator::Space => " ",
            PromptConcatenator::Comma => ", ",
            PromptConcatenator::Custom(custom) => custom,
        }
    }
}

pub trait PromptConcatenatorTrait {
    fn concate_mut(&mut self) -> &mut PromptConcatenator;

    fn clear_built(&mut self);

    fn concate_deol(&mut self) -> &mut Self {
        if self.concate_mut() != &PromptConcatenator::DoubleNewline {
            *self.concate_mut() = PromptConcatenator::DoubleNewline;
            self.clear_built();
        }
        self
    }

    fn concate_seol(&mut self) -> &mut Self {
        if self.concate_mut() != &PromptConcatenator::SingleNewline {
            *self.concate_mut() = PromptConcatenator::SingleNewline;
            self.clear_built();
        }
        self
    }

    fn concate_space(&mut self) -> &mut Self {
        if self.concate_mut() != &PromptConcatenator::Space {
            *self.concate_mut() = PromptConcatenator::Space;
            self.clear_built();
        }
        self
    }

    fn concate_comma(&mut self) -> &mut Self {
        if self.concate_mut() != &PromptConcatenator::Comma {
            *self.concate_mut() = PromptConcatenator::Comma;
            self.clear_built();
        }
        self
    }

    fn concate_custom<T: AsRef<str>>(&mut self, custom: T) -> &mut Self {
        if self.concate_mut() != &PromptConcatenator::Custom(custom.as_ref().to_owned()) {
            *self.concate_mut() = PromptConcatenator::Custom(custom.as_ref().to_owned());
            self.clear_built();
        }
        self
    }
}
