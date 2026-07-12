//! Apple BridgeSupport XML ingestion (`BridgeSupport.5`).

use quick_xml::events::{BytesEnd, BytesStart, Event};
use quick_xml::Reader;

use crate::model::{AbiArg, AbiFunction, AbiSpec, AbiValueKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeSupportError {
    Xml(String),
    UnexpectedRoot,
}

impl std::fmt::Display for BridgeSupportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Xml(e) => write!(f, "BridgeSupport XML: {e}"),
            Self::UnexpectedRoot => write!(f, "BridgeSupport XML: expected <signatures> root"),
        }
    }
}

impl std::error::Error for BridgeSupportError {}

/// Parse a BridgeSupport XML document into an [`AbiSpec`].
pub fn parse_bridgesupport(xml: &str) -> Result<AbiSpec, BridgeSupportError> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut spec = AbiSpec::default();
    let mut buf = Vec::new();
    let mut in_signatures = false;
    let mut cur_fn: Option<PartialFunction> = None;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let tag = tag_name(&e);
                if tag == "signatures" {
                    in_signatures = true;
                } else if in_signatures && tag == "function" {
                    cur_fn = Some(PartialFunction::new(&e)?);
                } else if let Some(ref mut pf) = cur_fn {
                    if tag == "arg" {
                        pf.args.push(parse_arg(&e)?);
                    } else if tag == "retval" {
                        pf.retval = Some(parse_arg(&e)?);
                    }
                }
            }
            Ok(Event::Empty(e)) => {
                let tag = tag_name(&e);
                if let Some(ref mut pf) = cur_fn {
                    if tag == "arg" {
                        pf.args.push(parse_arg(&e)?);
                    } else if tag == "retval" {
                        pf.retval = Some(parse_arg(&e)?);
                    }
                }
            }
            Ok(Event::End(e)) => {
                let tag = tag_name_end(&e);
                if tag == "function" {
                    if let Some(pf) = cur_fn.take() {
                        spec.functions.push(pf.finish());
                    }
                } else if tag == "signatures" {
                    in_signatures = false;
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(BridgeSupportError::Xml(e.to_string())),
            _ => {}
        }
        buf.clear();
    }

    if !in_signatures && spec.functions.is_empty() && !xml.contains("<signatures") {
        return Err(BridgeSupportError::UnexpectedRoot);
    }

    spec.functions.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(spec)
}

fn tag_name(e: &BytesStart<'_>) -> String {
    String::from_utf8_lossy(e.name().local_name().as_ref()).into_owned()
}

fn tag_name_end(e: &BytesEnd<'_>) -> String {
    String::from_utf8_lossy(e.name().local_name().as_ref()).into_owned()
}

fn attr(e: &BytesStart<'_>, key: &str) -> Option<String> {
    e.attributes()
        .filter_map(|a| a.ok())
        .find(|a| a.key.as_ref() == key.as_bytes())
        .map(|a| String::from_utf8_lossy(&a.value).into_owned())
}

fn parse_arg(e: &BytesStart<'_>) -> Result<AbiArg, BridgeSupportError> {
    let ty = attr(e, "type");
    let function_pointer = attr(e, "function_pointer")
        .map(|v| v == "true")
        .unwrap_or(false)
        || ty.as_deref() == Some("^?");
    let pointer = ty.as_deref().map(|t| t.starts_with('^') || t == "*").unwrap_or(false)
        || attr(e, "c_array_of_variable_length")
            .map(|v| v == "true")
            .unwrap_or(false);

    let kind = if function_pointer {
        AbiValueKind::FunctionPointer
    } else if ty.as_deref().map(|t| t.starts_with('@')).unwrap_or(false) {
        AbiValueKind::Object
    } else if pointer {
        AbiValueKind::Pointer
    } else if ty.as_deref() == Some("v") {
        AbiValueKind::Void
    } else if ty.is_some() {
        AbiValueKind::Scalar
    } else {
        AbiValueKind::Unknown(String::new())
    };

    Ok(AbiArg {
        kind,
        bridgesupport_type: ty,
        function_pointer,
        pointer,
    })
}

struct PartialFunction {
    name: String,
    variadic: bool,
    args: Vec<AbiArg>,
    retval: Option<AbiArg>,
}

impl PartialFunction {
    fn new(e: &BytesStart<'_>) -> Result<Self, BridgeSupportError> {
        let name = attr(e, "name").ok_or_else(|| {
            BridgeSupportError::Xml("function missing name attribute".into())
        })?;
        let variadic = attr(e, "variadic").map(|v| v == "true").unwrap_or(false);
        Ok(Self {
            name,
            variadic,
            args: Vec::new(),
            retval: None,
        })
    }

    fn finish(self) -> AbiFunction {
        AbiFunction {
            name: self.name,
            args: self.args,
            retval: self.retval.unwrap_or_else(AbiArg::void),
            variadic: self.variadic,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::AbiValueKind;

    const SAMPLE: &str = r#"<?xml version="1.0"?>
<signatures version="1.0">
  <function name="write">
    <arg type="i"/>
    <arg type="^v"/>
    <arg type="Q"/>
    <retval type="q"/>
  </function>
  <function name="printf">
    <arg type="*"/>
    <arg type="^?"/>
    <retval type="i"/>
  </function>
  <function name="exit">
    <arg type="i"/>
    <retval type="v"/>
  </function>
</signatures>"#;

    #[test]
    fn parses_functions_and_fn_ptr_args() {
        let spec = parse_bridgesupport(SAMPLE).unwrap();
        assert_eq!(spec.functions.len(), 3);

        let write = spec.lookup("write").unwrap();
        assert_eq!(write.args.len(), 3);
        assert_eq!(write.args[1].kind, AbiValueKind::Pointer);
        assert!(!write.retval.function_pointer);

        let printf = spec.lookup("printf").unwrap();
        assert!(printf.args[1].function_pointer);
        assert_eq!(printf.args[1].kind, AbiValueKind::FunctionPointer);
        assert!(spec.accepts_function_pointers("printf"));
        assert!(!spec.accepts_function_pointers("write"));
        assert_eq!(spec.fn_ptr_symbols(), vec!["printf"]);
    }

    #[test]
    fn extend_manifest_includes_fn_ptr_symbols_with_stub() {
        use speet_host_api::ImportManifest;

        let spec = parse_bridgesupport(SAMPLE).unwrap();
        let base = ImportManifest::native_syscall();
        let extended = AbiSpec::extend_manifest(&base, &spec, &["write", "printf", "exit"]);

        assert!(extended.index_of("env", "write").is_some());
        assert!(extended.index_of("env", "exit").is_some());
        assert!(extended.index_of("env", "printf").is_some());
    }
}
