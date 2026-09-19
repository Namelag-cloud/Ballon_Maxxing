from main import parse_args


def test_collect_mode_cli_parses():
    args = parse_args([
        "--mode", "collect",
        "--game", "balloons",
        "--fullscreen",
        "--startup-delay", "5",
        "--duration", "120",
    ])
    assert args.mode == "collect"
    assert args.game == "balloons"
    assert args.fullscreen is True
    assert args.startup_delay == 5.0
    assert args.duration == 120.0


def test_live_candidate_recording_cli_parses():
    args = parse_args([
        "--mode", "live",
        "--game", "balloons",
        "--record-candidates",
    ])
    assert args.mode == "live"
    assert args.game == "balloons"
    assert args.record_candidates is True
