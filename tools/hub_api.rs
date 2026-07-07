use forge_llm::mud::MudFile;
use serde_json::Value;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

fn handle_client(mut stream: TcpStream, mf_opt: Option<Arc<MudFile>>) {
    let mut buffer = [0; 4096];
    let bytes_read = match stream.read(&mut buffer) {
        Ok(n) if n > 0 => n,
        _ => return,
    };

    let request = String::from_utf8_lossy(&buffer[..bytes_read]);

    // Handle CORS Preflight
    if request.starts_with("OPTIONS ") {
        let response = "HTTP/1.1 204 NO CONTENT\r\n\
                        Access-Control-Allow-Origin: *\r\n\
                        Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n\
                        Access-Control-Allow-Headers: Content-Type, Authorization\r\n\
                        \r\n";
        let _ = stream.write_all(response.as_bytes());
        return;
    }

    if !request.starts_with("POST /chat/completions")
        && !request.starts_with("POST /v1/chat/completions")
    {
        let response = "HTTP/1.1 404 NOT FOUND\r\nContent-Length: 0\r\n\r\n";
        let _ = stream.write_all(response.as_bytes());
        return;
    }

    // Extract body
    if let Some(body_idx) = request.find("\r\n\r\n") {
        let body_str = &request[body_idx + 4..];

        let prompt_text = if let Ok(json) = serde_json::from_str::<Value>(body_str) {
            if let Some(msgs) = json["messages"].as_array() {
                // OpenAI Chat format compatibility
                msgs.last()
                    .and_then(|m| m["content"].as_str())
                    .unwrap_or("")
                    .to_string()
            } else {
                json["prompt"].as_str().unwrap_or("").to_string()
            }
        } else {
            "".to_string()
        };

        println!("Received prompt: {}", prompt_text);

        // Streaming Response headers
        let response_headers = "HTTP/1.1 200 OK\r\n\
                                Content-Type: text/event-stream\r\n\
                                Cache-Control: no-cache\r\n\
                                Connection: keep-alive\r\n\
                                Access-Control-Allow-Origin: *\r\n\
                                \r\n";
        if stream.write_all(response_headers.as_bytes()).is_err() {
            return;
        }

        // Setup SlimeWorkspace dimensions
        let mut hidden = 1024;
        let mut max_pos = 128;
        let mut n_heads = 8;
        let mut head_dim = 128;

        if let Some(mf) = &mf_opt {
            if let Some(h) = mf
                .global_metadata
                .get("hidden_size")
                .and_then(|s| s.parse::<usize>().ok())
            {
                hidden = h;
            }
            if let Some(m) = mf
                .global_metadata
                .get("max_position_embeddings")
                .and_then(|s| s.parse::<usize>().ok())
            {
                max_pos = m;
            }
            if let Some(n) = mf
                .global_metadata
                .get("num_heads")
                .and_then(|s| s.parse::<usize>().ok())
            {
                n_heads = n;
            }
            if let Some(d) = mf
                .global_metadata
                .get("head_dim")
                .and_then(|s| s.parse::<usize>().ok())
            {
                head_dim = d;
            }

            if let Some(raw_cfg) = mf.raw_config() {
                if hidden == 1024 {
                    if let Some(h) = raw_cfg.get("hidden_size").and_then(|v| v.as_u64()) {
                        hidden = h as usize;
                    }
                }
            }
        }

        let mut ws = forge_llm::mud::slime::SlimeWorkspace::new(
            hidden, max_pos, n_heads, n_heads, head_dim, hidden, 30, 128.0,
        );

        // For this API stub we mock the layer pointers as done in main.rs
        let mock_weights = vec![0x11u8; hidden * hidden / 2];
        let mock_scales = vec![0.01f32; hidden];
        let mock_norm = vec![1.0f32; hidden];

        let layer = forge_llm::mud::slime_forward::SlimeLayer {
            q_w: mock_weights.as_ptr(),
            k_w: mock_weights.as_ptr(),
            v_w: mock_weights.as_ptr(),
            o_w: mock_weights.as_ptr(),
            q_scales: mock_scales.as_ptr(),
            k_scales: mock_scales.as_ptr(),
            v_scales: mock_scales.as_ptr(),
            o_scales: mock_scales.as_ptr(),
            ffn_up_w: mock_weights.as_ptr(),
            ffn_gate_w: mock_weights.as_ptr(),
            ffn_down_w: mock_weights.as_ptr(),
            ffn_up_scales: mock_scales.as_ptr(),
            ffn_gate_scales: mock_scales.as_ptr(),
            ffn_down_scales: mock_scales.as_ptr(),
            attn_norm_w: mock_norm.as_ptr(),
            ffn_norm_w: mock_norm.as_ptr(),
            attn_sub_norm_w: std::ptr::null(),
            ffn_sub_norm_w: std::ptr::null(),
            mhc_alpha_w: std::ptr::null(),
            mhc_beta_w: std::ptr::null(),
            mhc_radius_w: std::ptr::null(),
            n_kv_heads: n_heads,
            ffn_mid: hidden,
            rope_theta: 0.0,
        };

        let mut stream_clone = stream.try_clone().unwrap();

        // Mock token generation loop using SlimeBlock evaluations
        let mock_response = vec![
            " Hello", " world,", " this", " is", " the", " MUD", " Hub", " &", " Spoke", " API!",
        ];

        for (i, word) in mock_response.iter().enumerate() {
            forge_llm::mud::slime_forward::evaluate_slime_block(&layer, 0, &mut ws, i, 1e-6, None);

            let chunk = serde_json::json!({
                "id": "chatcmpl-mud123",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": { "content": word },
                    "finish_reason": serde_json::Value::Null
                }]
            });
            let data = format!("data: {}\n\n", chunk);
            let _ = stream_clone.write_all(data.as_bytes());
            let _ = stream_clone.flush();
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        let chunk = serde_json::json!({
            "id": "chatcmpl-mud123",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        });
        let _ = stream.write_all(format!("data: {}\n\n", chunk).as_bytes());
        let _ = stream.write_all("data: [DONE]\n\n".as_bytes());
    }
}

fn main() -> anyhow::Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/core_skills.mud".to_string());

    println!("=== MUD Local Hub & Spoke API ===");
    println!("Loading Model: {}", model_path);

    let mud_file = if let Ok(exe_path) = std::env::current_exe() {
        if let Ok(mf) = MudFile::load(exe_path.to_str().unwrap()) {
            println!("MUD payload loaded from current executable.");
            Some(Arc::new(mf))
        } else {
            MudFile::load(&model_path).ok().map(Arc::new)
        }
    } else {
        MudFile::load(&model_path).ok().map(Arc::new)
    };

    if mud_file.is_none() {
        println!("Warning: No valid .mud file loaded. Running in mock fallback mode.");
    }

    let listener = TcpListener::bind("0.0.0.0:8080")?;
    println!("Hub API listening on http://0.0.0.0:8080");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let mf_clone = mud_file.clone();
                std::thread::spawn(move || {
                    handle_client(stream, mf_clone);
                });
            }
            Err(e) => eprintln!("Failed to establish connection: {}", e),
        }
    }

    Ok(())
}
