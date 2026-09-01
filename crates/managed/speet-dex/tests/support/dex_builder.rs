//! Minimal, toolchain-free DEX (Dalvik Executable) container encoder.
//!
//! Emits just enough of the real DEX file format (see
//! <https://source.android.com/docs/core/runtime/dex-format>) for the vendored
//! `dex` (dex-parser) crate to parse successfully. Deliberately narrow:
//! no static fields, no annotations, no debug info, no interfaces, no tries.
//! Field/method IDs within a single class's field/method lists must be added
//! in ascending order (DEX's diff-encoding requires non-decreasing IDs).

#![allow(dead_code)]

fn adler32(data: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65521;
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }
    (b << 16) | a
}

fn pad4(buf: &mut Vec<u8>) {
    while buf.len() % 4 != 0 {
        buf.push(0);
    }
}

#[derive(Clone)]
pub struct ProtoDef {
    pub shorty_str: u32,
    pub return_type: u32,
    pub params: Vec<u32>,
}

#[derive(Clone)]
pub struct FieldIdDef {
    pub class_type: u32,
    pub field_type: u32,
    pub name_str: u32,
}

#[derive(Clone)]
pub struct MethodIdDef {
    pub class_type: u32,
    pub proto: u32,
    pub name_str: u32,
}

#[derive(Clone)]
pub struct CodeDef {
    pub registers_size: u16,
    pub ins_size: u16,
    pub outs_size: u16,
    pub insns: Vec<u16>,
}

#[derive(Clone)]
pub struct EncodedMemberDef {
    pub id: u32,
    pub access_flags: u32,
    pub code: Option<CodeDef>,
}

#[derive(Clone, Default)]
pub struct ClassDef {
    pub class_type: u32,
    pub access_flags: u32,
    pub instance_fields: Vec<EncodedMemberDef>,
    pub direct_methods: Vec<EncodedMemberDef>,
    pub virtual_methods: Vec<EncodedMemberDef>,
}

#[derive(Default)]
pub struct DexBuilder {
    strings: Vec<String>,
    /// type_id -> string_id of the descriptor.
    types: Vec<u32>,
    protos: Vec<ProtoDef>,
    fields: Vec<FieldIdDef>,
    methods: Vec<MethodIdDef>,
    classes: Vec<ClassDef>,
}

impl DexBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_string(&mut self, s: &str) -> u32 {
        if let Some(pos) = self.strings.iter().position(|existing| existing == s) {
            return pos as u32;
        }
        self.strings.push(s.to_string());
        (self.strings.len() - 1) as u32
    }

    /// Add a type by its DEX descriptor (e.g. `"LFoo;"`, `"I"`), returning its type_id.
    pub fn add_type(&mut self, descriptor: &str) -> u32 {
        let str_id = self.add_string(descriptor);
        if let Some(pos) = self.types.iter().position(|&s| s == str_id) {
            return pos as u32;
        }
        self.types.push(str_id);
        (self.types.len() - 1) as u32
    }

    pub fn add_proto(&mut self, shorty: &str, return_desc: &str, param_descs: &[&str]) -> u32 {
        let shorty_str = self.add_string(shorty);
        let return_type = self.add_type(return_desc);
        let params = param_descs.iter().map(|d| self.add_type(d)).collect();
        self.protos.push(ProtoDef { shorty_str, return_type, params });
        (self.protos.len() - 1) as u32
    }

    pub fn add_field(&mut self, class_desc: &str, field_desc: &str, name: &str) -> u32 {
        let class_type = self.add_type(class_desc);
        let field_type = self.add_type(field_desc);
        let name_str = self.add_string(name);
        self.fields.push(FieldIdDef { class_type, field_type, name_str });
        (self.fields.len() - 1) as u32
    }

    pub fn add_method(&mut self, class_desc: &str, proto: u32, name: &str) -> u32 {
        let class_type = self.add_type(class_desc);
        let name_str = self.add_string(name);
        self.methods.push(MethodIdDef { class_type, proto, name_str });
        (self.methods.len() - 1) as u32
    }

    pub fn add_class(&mut self, class: ClassDef) {
        self.classes.push(class);
    }

    pub fn build(&self) -> Vec<u8> {
        let string_ids_size = self.strings.len() as u32;
        let type_ids_size = self.types.len() as u32;
        let proto_ids_size = self.protos.len() as u32;
        let field_ids_size = self.fields.len() as u32;
        let method_ids_size = self.methods.len() as u32;
        let class_defs_size = self.classes.len() as u32;

        const HEADER_SIZE: u32 = 0x70;
        let string_ids_off = HEADER_SIZE;
        let type_ids_off = string_ids_off + string_ids_size * 4;
        let proto_ids_off = type_ids_off + type_ids_size * 4;
        let field_ids_off = proto_ids_off + proto_ids_size * 12;
        let method_ids_off = field_ids_off + field_ids_size * 8;
        let class_defs_off = method_ids_off + method_ids_size * 8;
        let data_off = class_defs_off + class_defs_size * 32;

        // ── Data section ────────────────────────────────────────────────
        let mut data: Vec<u8> = Vec::new();

        // string_data_item* (ASCII only: utf16_size == byte length, MUTF-8 == ASCII).
        let mut string_data_offsets = Vec::with_capacity(self.strings.len());
        for s in &self.strings {
            string_data_offsets.push(data_off + data.len() as u32);
            write_uleb128(&mut data, s.len() as u64);
            data.extend_from_slice(s.as_bytes());
            data.push(0);
        }

        // type_list per proto with non-empty params (4-byte aligned).
        let mut proto_params_offsets = Vec::with_capacity(self.protos.len());
        for p in &self.protos {
            if p.params.is_empty() {
                proto_params_offsets.push(0u32);
                continue;
            }
            pad4(&mut data);
            proto_params_offsets.push(data_off + data.len() as u32);
            data.extend_from_slice(&(p.params.len() as u32).to_le_bytes());
            for &t in &p.params {
                data.extend_from_slice(&(t as u16).to_le_bytes());
            }
        }

        // code_item* (4-byte aligned), recording each method's offset by (class idx, member idx, is_direct).
        struct CodeOffsetKey {
            class_idx: usize,
            direct: bool,
            member_idx: usize,
        }
        let mut code_offsets: Vec<(CodeOffsetKey, u32)> = Vec::new();
        for (class_idx, class) in self.classes.iter().enumerate() {
            for (member_idx, m) in class.direct_methods.iter().enumerate() {
                if let Some(code) = &m.code {
                    pad4(&mut data);
                    let off = data_off + data.len() as u32;
                    code_offsets.push((CodeOffsetKey { class_idx, direct: true, member_idx }, off));
                    write_code_item(&mut data, code);
                }
            }
            for (member_idx, m) in class.virtual_methods.iter().enumerate() {
                if let Some(code) = &m.code {
                    pad4(&mut data);
                    let off = data_off + data.len() as u32;
                    code_offsets.push((CodeOffsetKey { class_idx, direct: false, member_idx }, off));
                    write_code_item(&mut data, code);
                }
            }
        }
        let code_off_for = |class_idx: usize, direct: bool, member_idx: usize| -> u32 {
            code_offsets
                .iter()
                .find(|(k, _)| k.class_idx == class_idx && k.direct == direct && k.member_idx == member_idx)
                .map(|(_, off)| *off)
                .unwrap_or(0)
        };

        // class_data_item* — one per class that has any fields/methods.
        let mut class_data_offsets = vec![0u32; self.classes.len()];
        for (class_idx, class) in self.classes.iter().enumerate() {
            if class.instance_fields.is_empty()
                && class.direct_methods.is_empty()
                && class.virtual_methods.is_empty()
            {
                continue;
            }
            class_data_offsets[class_idx] = data_off + data.len() as u32;
            write_uleb128(&mut data, 0); // static_fields_size
            write_uleb128(&mut data, class.instance_fields.len() as u64);
            write_uleb128(&mut data, class.direct_methods.len() as u64);
            write_uleb128(&mut data, class.virtual_methods.len() as u64);

            let mut prev = 0u32;
            for f in &class.instance_fields {
                debug_assert!(f.id >= prev, "instance fields must be added in ascending field_id order");
                write_uleb128(&mut data, (f.id - prev) as u64);
                write_uleb128(&mut data, f.access_flags as u64);
                prev = f.id;
            }
            let mut prev = 0u32;
            for (member_idx, m) in class.direct_methods.iter().enumerate() {
                debug_assert!(m.id >= prev, "direct methods must be added in ascending method_id order");
                write_uleb128(&mut data, (m.id - prev) as u64);
                write_uleb128(&mut data, m.access_flags as u64);
                write_uleb128(&mut data, code_off_for(class_idx, true, member_idx) as u64);
                prev = m.id;
            }
            let mut prev = 0u32;
            for (member_idx, m) in class.virtual_methods.iter().enumerate() {
                debug_assert!(m.id >= prev, "virtual methods must be added in ascending method_id order");
                write_uleb128(&mut data, (m.id - prev) as u64);
                write_uleb128(&mut data, m.access_flags as u64);
                write_uleb128(&mut data, code_off_for(class_idx, false, member_idx) as u64);
                prev = m.id;
            }
        }

        // map_list (4-byte aligned) — last item in the data section.
        pad4(&mut data);
        let map_off = data_off + data.len() as u32;
        {
            let mut items: Vec<(u16, u32, u32)> = Vec::new(); // (item_type, size, offset)
            items.push((0x0000, 1, 0)); // Header
            if string_ids_size > 0 { items.push((0x0001, string_ids_size, string_ids_off)); }
            if type_ids_size > 0 { items.push((0x0002, type_ids_size, type_ids_off)); }
            if proto_ids_size > 0 { items.push((0x0003, proto_ids_size, proto_ids_off)); }
            if field_ids_size > 0 { items.push((0x0004, field_ids_size, field_ids_off)); }
            if method_ids_size > 0 { items.push((0x0005, method_ids_size, method_ids_off)); }
            if class_defs_size > 0 { items.push((0x0006, class_defs_size, class_defs_off)); }
            let n_type_lists = proto_params_offsets.iter().filter(|&&o| o != 0).count() as u32;
            if n_type_lists > 0 {
                let first = *proto_params_offsets.iter().find(|&&o| o != 0).unwrap();
                items.push((0x1001, n_type_lists, first));
            }
            let n_class_data = class_data_offsets.iter().filter(|&&o| o != 0).count() as u32;
            if n_class_data > 0 {
                let first = *class_data_offsets.iter().find(|&&o| o != 0).unwrap();
                items.push((0x2000, n_class_data, first));
            }
            let n_code_items = code_offsets.len() as u32;
            if n_code_items > 0 {
                let first = code_offsets.iter().map(|(_, off)| *off).min().unwrap();
                items.push((0x2001, n_code_items, first));
            }
            if string_ids_size > 0 {
                items.push((0x2002, string_ids_size, string_data_offsets[0]));
            }
            items.push((0x1000, 1, map_off)); // MapList itself
            items.sort_by_key(|(_, _, off)| *off);

            data.extend_from_slice(&(items.len() as u32).to_le_bytes());
            for (item_type, size, offset) in items {
                data.extend_from_slice(&item_type.to_le_bytes());
                data.extend_from_slice(&0u16.to_le_bytes());
                data.extend_from_slice(&size.to_le_bytes());
                data.extend_from_slice(&offset.to_le_bytes());
            }
        }

        pad4(&mut data);
        let data_size = data.len() as u32;
        let file_size = data_off + data_size;

        // ── Assemble full file ──────────────────────────────────────────
        let mut out: Vec<u8> = Vec::with_capacity(file_size as usize);
        out.extend_from_slice(b"dex\n035\0"); // magic
        out.extend_from_slice(&0u32.to_le_bytes()); // checksum placeholder
        out.extend_from_slice(&[0u8; 20]); // sha1 signature (unchecked by dex-parser)
        out.extend_from_slice(&file_size.to_le_bytes());
        out.extend_from_slice(&HEADER_SIZE.to_le_bytes());
        out.extend_from_slice(&[0x78, 0x56, 0x34, 0x12]); // endian_tag (little-endian)
        out.extend_from_slice(&0u32.to_le_bytes()); // link_size
        out.extend_from_slice(&0u32.to_le_bytes()); // link_off
        out.extend_from_slice(&map_off.to_le_bytes());
        out.extend_from_slice(&string_ids_size.to_le_bytes());
        out.extend_from_slice(&string_ids_off.to_le_bytes());
        out.extend_from_slice(&type_ids_size.to_le_bytes());
        out.extend_from_slice(&type_ids_off.to_le_bytes());
        out.extend_from_slice(&proto_ids_size.to_le_bytes());
        out.extend_from_slice(&proto_ids_off.to_le_bytes());
        out.extend_from_slice(&field_ids_size.to_le_bytes());
        out.extend_from_slice(&field_ids_off.to_le_bytes());
        out.extend_from_slice(&method_ids_size.to_le_bytes());
        out.extend_from_slice(&method_ids_off.to_le_bytes());
        out.extend_from_slice(&class_defs_size.to_le_bytes());
        out.extend_from_slice(&class_defs_off.to_le_bytes());
        out.extend_from_slice(&data_size.to_le_bytes());
        out.extend_from_slice(&data_off.to_le_bytes());
        assert_eq!(out.len() as u32, HEADER_SIZE);

        for &off in &string_data_offsets {
            out.extend_from_slice(&off.to_le_bytes());
        }
        for &str_id in &self.types {
            out.extend_from_slice(&str_id.to_le_bytes());
        }
        for (idx, p) in self.protos.iter().enumerate() {
            out.extend_from_slice(&p.shorty_str.to_le_bytes());
            out.extend_from_slice(&p.return_type.to_le_bytes());
            let params_off = proto_params_offsets.get(idx).copied().unwrap_or(0);
            out.extend_from_slice(&params_off.to_le_bytes());
        }
        for f in &self.fields {
            out.extend_from_slice(&(f.class_type as u16).to_le_bytes());
            out.extend_from_slice(&(f.field_type as u16).to_le_bytes());
            out.extend_from_slice(&f.name_str.to_le_bytes());
        }
        for m in &self.methods {
            out.extend_from_slice(&(m.class_type as u16).to_le_bytes());
            out.extend_from_slice(&(m.proto as u16).to_le_bytes());
            out.extend_from_slice(&m.name_str.to_le_bytes());
        }
        for (class_idx, class) in self.classes.iter().enumerate() {
            out.extend_from_slice(&class.class_type.to_le_bytes());
            out.extend_from_slice(&class.access_flags.to_le_bytes());
            out.extend_from_slice(&0xffff_ffffu32.to_le_bytes()); // superclass_idx = NO_INDEX
            out.extend_from_slice(&0u32.to_le_bytes()); // interfaces_off
            out.extend_from_slice(&0xffff_ffffu32.to_le_bytes()); // source_file_idx = NO_INDEX
            out.extend_from_slice(&0u32.to_le_bytes()); // annotations_off
            out.extend_from_slice(&class_data_offsets[class_idx].to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes()); // static_values_off
        }

        assert_eq!(out.len() as u32, data_off);
        out.extend_from_slice(&data);
        assert_eq!(out.len() as u32, file_size);

        let checksum = adler32(&out[12..]);
        out[8..12].copy_from_slice(&checksum.to_le_bytes());

        out
    }
}

fn write_code_item(buf: &mut Vec<u8>, code: &CodeDef) {
    buf.extend_from_slice(&code.registers_size.to_le_bytes());
    buf.extend_from_slice(&code.ins_size.to_le_bytes());
    buf.extend_from_slice(&code.outs_size.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes()); // tries_size
    buf.extend_from_slice(&0u32.to_le_bytes()); // debug_info_off
    buf.extend_from_slice(&(code.insns.len() as u32).to_le_bytes());
    for unit in &code.insns {
        buf.extend_from_slice(&unit.to_le_bytes());
    }
}

fn write_uleb128(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if value == 0 {
            break;
        }
    }
}
