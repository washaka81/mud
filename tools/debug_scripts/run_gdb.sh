#!/bin/bash
cargo build --bin universal_converter --features="tools"
gdb -ex 'run models/bitnet-b1.58-2B-4T/model.safetensors models/bitnet-b1.58-2B-4T.mud' -ex 'bt' target/debug/universal_converter < /dev/null
