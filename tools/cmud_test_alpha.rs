use forge_llm::mud::cmud::CmudLayerParams;
use forge_llm::mud::MudFile;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mud_path = &args[1];
    let mut mud = MudFile::load(mud_path).unwrap();
    mud.remap_tensors().unwrap();
    let hidden = mud.meta.hidden_size as usize;
    let seqs_data = vec![(vec![100, 200], 300)]; // just 1 short seq

    let tmp = std::env::temp_dir().join("cmud_test_alpha.json");
    std::env::set_var("MUD_CMUD_PARAMS", &tmp);

    let params = CmudLayerParams::from_defaults(hidden);
    params.save_json(&tmp).ok();

    let (_, grad, _) = forge_llm::mud::inference::cmud_training_forward(&mud, &seqs_data[0].0, seqs_data[0].1).unwrap();
    let an = grad.alpha;

    let loss_of = |mut p: CmudLayerParams, delta: f32| -> f32 {
        p.alpha += delta;
        p.save_json(&tmp).ok();
        forge_llm::mud::inference::cmud_training_forward(&mud, &seqs_data[0].0, seqs_data[0].1)
            .unwrap()
            .0
    };

    let eps = 1e-3;
    let fd = (loss_of(params.clone(), eps) - loss_of(params.clone(), -eps)) / (2.0 * eps);

    println!("alpha an={an:.5} fd={fd:.5}");
    ExitCode::SUCCESS
}
