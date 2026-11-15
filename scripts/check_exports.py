#!/usr/bin/env python3
"""Check which symbols a built plugin actually exports.

    python scripts/check_exports.py dist/ets2ad_telemetry.so \\
        scs_telemetry_init scs_telemetry_shutdown scs_input_init scs_input_shutdown

The game loads the plugin and looks up these four names. If they are not
exported it loads the library, finds nothing, and carries on without a word --
no error, no log line, no telemetry. That failure is silent on both platforms
and was real on Windows until `exports.def` was added, so it gets a check.

`--exact` additionally requires that *nothing else* is exported. The game
`dlopen`s this into its own process, where a stray symbol can collide with the
game's own or another plugin's, and those failures look like game bugs.

The export tables are parsed directly rather than shelled out to `nm` or
`dumpbin`. `nm` is not on Windows, `dumpbin` needs a Developer Command Prompt
that a plain shell step does not have, and a check that only runs on one
platform is not much of a check. Both formats are a few dozen lines.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

__all__ = ["exported_symbols"]

# ELF constants.
_SHT_DYNSYM = 11
_STB_GLOBAL = 1
_STT_FUNC = 2
_SHN_UNDEF = 0


class UnsupportedBinary(ValueError):
    """Not a format this script knows how to read."""


def exported_symbols(path: Path) -> set[str]:
    """Return the names a dynamic linker can resolve in this binary."""
    data = path.read_bytes()
    if data[:4] == b"\x7fELF":
        return _elf_exports(data)
    if data[:2] == b"MZ":
        return _pe_exports(data)
    raise UnsupportedBinary(f"{path} is neither ELF nor PE")


# --------------------------------------------------------------------------- #
# ELF
# --------------------------------------------------------------------------- #


def _elf_exports(data: bytes) -> set[str]:
    if data[4] != 2:  # EI_CLASS
        raise UnsupportedBinary("only 64-bit ELF is supported; the game is x64 only")
    if data[5] != 1:  # EI_DATA
        raise UnsupportedBinary("only little-endian ELF is supported")

    (e_shoff,) = struct.unpack_from("<Q", data, 0x28)
    e_shentsize, e_shnum = struct.unpack_from("<HH", data, 0x3A)

    sections = []
    for index in range(e_shnum):
        base = e_shoff + index * e_shentsize
        (sh_type,) = struct.unpack_from("<I", data, base + 4)
        sh_offset, sh_size = struct.unpack_from("<QQ", data, base + 0x18)
        (sh_link,) = struct.unpack_from("<I", data, base + 0x28)
        (sh_entsize,) = struct.unpack_from("<Q", data, base + 0x38)
        sections.append((sh_type, sh_offset, sh_size, sh_link, sh_entsize))

    for sh_type, sh_offset, sh_size, sh_link, sh_entsize in sections:
        if sh_type != _SHT_DYNSYM or sh_entsize == 0:
            continue
        _, str_offset, str_size, _, _ = sections[sh_link]
        strings = data[str_offset : str_offset + str_size]

        names: set[str] = set()
        for entry in range(sh_size // sh_entsize):
            base = sh_offset + entry * sh_entsize
            st_name, st_info, _st_other, st_shndx = struct.unpack_from("<IBBH", data, base)
            if st_shndx == _SHN_UNDEF:
                continue  # imported, not exported
            if (st_info >> 4) != _STB_GLOBAL or (st_info & 0xF) != _STT_FUNC:
                continue
            end = strings.index(b"\x00", st_name)
            names.add(strings[st_name:end].decode("ascii"))
        return names

    return set()


# --------------------------------------------------------------------------- #
# PE
# --------------------------------------------------------------------------- #


def _pe_exports(data: bytes) -> set[str]:
    (e_lfanew,) = struct.unpack_from("<I", data, 0x3C)
    if data[e_lfanew : e_lfanew + 4] != b"PE\x00\x00":
        raise UnsupportedBinary("missing PE signature")

    coff = e_lfanew + 4
    (size_of_optional,) = struct.unpack_from("<H", data, coff + 16)
    (number_of_sections,) = struct.unpack_from("<H", data, coff + 2)
    optional = coff + 20

    (magic,) = struct.unpack_from("<H", data, optional)
    if magic == 0x20B:  # PE32+
        directories = optional + 112
    elif magic == 0x10B:  # PE32
        directories = optional + 96
    else:
        raise UnsupportedBinary(f"unknown optional header magic 0x{magic:04X}")

    export_rva, export_size = struct.unpack_from("<II", data, directories)
    if export_rva == 0 or export_size == 0:
        return set()  # a DLL with no exports at all -- the failure this catches

    sections = []
    base = optional + size_of_optional
    for index in range(number_of_sections):
        header = base + index * 40
        virtual_size, virtual_address = struct.unpack_from("<II", data, header + 8)
        raw_size, raw_pointer = struct.unpack_from("<II", data, header + 16)
        sections.append((virtual_address, max(virtual_size, raw_size), raw_pointer))

    def to_offset(rva: int) -> int:
        for virtual_address, size, raw_pointer in sections:
            if virtual_address <= rva < virtual_address + size:
                return raw_pointer + (rva - virtual_address)
        raise UnsupportedBinary(f"RVA 0x{rva:X} is outside every section")

    directory = to_offset(export_rva)
    (number_of_names,) = struct.unpack_from("<I", data, directory + 24)
    (names_rva,) = struct.unpack_from("<I", data, directory + 32)
    names_table = to_offset(names_rva)

    names: set[str] = set()
    for index in range(number_of_names):
        (name_rva,) = struct.unpack_from("<I", data, names_table + index * 4)
        offset = to_offset(name_rva)
        end = data.index(b"\x00", offset)
        names.add(data[offset:end].decode("ascii"))
    return names


# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("expected", nargs="+", help="symbols that must be exported")
    parser.add_argument(
        "--exact", action="store_true", help="also fail if anything else is exported"
    )
    args = parser.parse_args()

    try:
        found = exported_symbols(args.binary)
    except (UnsupportedBinary, OSError, IndexError, struct.error) as exc:
        print(f"could not read {args.binary}: {exc}", file=sys.stderr)
        return 2

    expected = set(args.expected)
    missing = sorted(expected - found)
    extra = sorted(found - expected)

    for symbol in sorted(found):
        print(f"  {symbol}")

    if missing:
        print(f"\nMISSING EXPORTS: {missing}", file=sys.stderr)
        print(
            "The game will load this library, find nothing, and say nothing.",
            file=sys.stderr,
        )
        return 1
    if args.exact and extra:
        print(f"\nUNEXPECTED EXPORTS: {extra}", file=sys.stderr)
        print(
            "The game dlopens this into its own process; stray symbols there "
            "collide in ways that look like game bugs.",
            file=sys.stderr,
        )
        return 1

    print(f"\n{len(found)} exports, all expected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
