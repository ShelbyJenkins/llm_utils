use gbnf::*;

#[derive(PartialEq, Clone)]
pub enum RestrictedCharacterSet {
    AlphaLower,
    AlphaUpper,
    Numeric,
    PunctuationGrammar,
    PunctuationExtended,
}
impl RestrictedCharacterSet {
    pub fn get_all_variants() -> Vec<RestrictedCharacterSet> {
        vec![
            RestrictedCharacterSet::AlphaLower,
            RestrictedCharacterSet::AlphaUpper,
            RestrictedCharacterSet::Numeric,
            RestrictedCharacterSet::PunctuationGrammar,
            RestrictedCharacterSet::PunctuationExtended,
        ]
    }
}

/// Text Structured generator
///
pub fn create_text_structured_grammar(restricted: Vec<RestrictedCharacterSet>) -> String {
    Grammar {
        items: vec![
            GrammarItem::Rule(Rule {
                lhs: NonTerminalSymbol {
                    name: "root".to_string(),
                },
                rhs: Production {
                    items: vec![ProductionItem::NonTerminal(
                        NonTerminalSymbol {
                            name: "char+".to_string(),
                        },
                        RepetitionType::One,
                    )],
                },
            }),
            GrammarItem::Rule(Rule {
                lhs: NonTerminalSymbol {
                    name: "char".to_string(),
                },
                rhs: Production {
                    items: vec![build_restricted_char_set(restricted)],
                },
            }),
        ],
    }
    .to_string()
}

fn build_restricted_char_set(restricted: Vec<RestrictedCharacterSet>) -> ProductionItem {
    let mut items = Vec::new();

    for set in restricted {
        match set {
            RestrictedCharacterSet::AlphaLower => {
                items.push(CharacterSetItem::CharacterRange('a', 'z'))
            }
            RestrictedCharacterSet::AlphaUpper => {
                items.push(CharacterSetItem::CharacterRange('A', 'Z'))
            }
            RestrictedCharacterSet::Numeric => {
                items.push(CharacterSetItem::CharacterRange('0', '9'))
            }
            RestrictedCharacterSet::PunctuationGrammar => {
                items.push(CharacterSetItem::Character('.'));
                items.push(CharacterSetItem::Character('!'));
                items.push(CharacterSetItem::Character('?'));
                items.push(CharacterSetItem::Character(','));
                items.push(CharacterSetItem::Character('\''));
                items.push(CharacterSetItem::Character('\"'));
                items.push(CharacterSetItem::Character(';'));
                items.push(CharacterSetItem::Character(':'));
            }
            RestrictedCharacterSet::PunctuationExtended => {
                items.push(CharacterSetItem::Character('-'));
                items.push(CharacterSetItem::Character('{'));
                items.push(CharacterSetItem::Character('}'));
                items.push(CharacterSetItem::Character('('));
                items.push(CharacterSetItem::Character(')'));
                items.push(CharacterSetItem::Character('<'));
                items.push(CharacterSetItem::Character('>'));
                items.push(CharacterSetItem::Character('@'));
                items.push(CharacterSetItem::Character('#'));
                items.push(CharacterSetItem::Character('$'));
                items.push(CharacterSetItem::Character('%'));
                items.push(CharacterSetItem::Character('^'));
                items.push(CharacterSetItem::Character('&'));
                items.push(CharacterSetItem::Character('*'));
                items.push(CharacterSetItem::Character('+'));
                items.push(CharacterSetItem::Character('='));
                items.push(CharacterSetItem::Character('~')); // This one was a problem. Maybe.
                items.push(CharacterSetItem::Character('|'));
                items.push(CharacterSetItem::Character('/'));
                items.push(CharacterSetItem::Backslash);
                items.push(CharacterSetItem::Hex("5B".to_string())); // [
                items.push(CharacterSetItem::Hex("5D".to_string())); // ]
            }
        }
    }
    // items.push(CharacterSetItem::Character(' '));
    ProductionItem::CharacterSet(
        CharacterSet {
            is_complement: true,
            items,
        },
        RepetitionType::One,
    )
}

/// Text list generator
///
pub fn create_list_grammar(min_items: u16, max_items: u16) -> String {
    let g = Grammar {
        items: vec![
            GrammarItem::Rule(Rule {
                lhs: NonTerminalSymbol {
                    name: "root".to_string(),
                },
                rhs: Production {
                    items: vec![ProductionItem::NonTerminal(
                        NonTerminalSymbol {
                            name: "patch_out_item".to_string(),
                        },
                        RepetitionType::One,
                    )],
                },
            }),
            GrammarItem::Rule(Rule {
                lhs: NonTerminalSymbol {
                    name: "item".to_string(),
                },
                rhs: Production {
                    items: vec![
                        ProductionItem::Terminal(
                            TerminalSymbol {
                                value: "- ".to_string(),
                            },
                            RepetitionType::One,
                        ),
                        build_removed_character_set(),
                        ProductionItem::Terminal(
                            TerminalSymbol {
                                value: "\\n".to_string(),
                            },
                            RepetitionType::One,
                        ),
                    ],
                },
            }),
        ],
    }
    .to_string();
    g.replace(
        "patch_out_item",
        &build_patched_list_frequency(min_items, max_items),
    )
}

fn build_removed_character_set() -> ProductionItem {
    ProductionItem::CharacterSet(
        CharacterSet {
            is_complement: true,
            items: vec![
                CharacterSetItem::Return,
                CharacterSetItem::NewLine,
                CharacterSetItem::Hex("0b".to_string()),
                CharacterSetItem::Hex("0c".to_string()),
                CharacterSetItem::Hex("85".to_string()),
                CharacterSetItem::Unicode("2028".to_string()),
                CharacterSetItem::Unicode("2029".to_string()),
            ],
        },
        RepetitionType::OneOrMore,
    )
}

/// This isn't currently used because the grammar crate doesn't support nesting as required
/// Instead we use build_patched_item_frequency to build the rule as a string
// fn build_list_frequency(&self) -> Vec<ProductionItem> {
//     let mut items = vec![];
//     let optional_count = cmp::max(self.max_items, self.min_items) - self.min_items;

//     for _ in 0..self.min_items {
//         items.push(ProductionItem::NonTerminal(
//             NonTerminalSymbol {
//                 name: "item".to_string(),
//             },
//             RepetitionType::One,
//         ))
//     }
//     for _ in 0..optional_count {
//         items.push(ProductionItem::NonTerminal(
//             NonTerminalSymbol {
//                 name: "item".to_string(),
//             },
//             RepetitionType::ZeroOrOne,
//         ))
//     }
//     items
// }

fn build_patched_list_frequency(min_items: u16, max_items: u16) -> String {
    let mut item_rule = String::from("");
    let optional_count = std::cmp::max(max_items, min_items) - min_items;

    for _ in 0..min_items {
        item_rule.push_str("item ");
    }
    for _ in 0..optional_count {
        item_rule.push_str("(item");
    }
    for _ in 0..optional_count {
        item_rule.push_str(")?");
    }

    item_rule
}
