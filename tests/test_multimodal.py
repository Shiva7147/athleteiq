"""
tests/test_multimodal.py
Unit tests verifying the morning audio briefing synthesizer and tactical video storyboard engine.
"""
import os
import pytest
from multimodal.audio_briefing import MorningAudioBriefingSynthesizer
from multimodal.video_breakdown import TacticalVideoGenerator
from agent.tools import get_risk_ranking


def test_audio_briefing_script_composition():
    """Verifies that the audio script includes the 3-alert budget and plain-English directives."""
    synthesizer = MorningAudioBriefingSynthesizer(squad_name="Arsenal First Team")
    athletes = get_risk_ranking(top_n=3)
    
    briefing = synthesizer.compose_briefing_script(athletes, match_days_away=2)
    assert briefing.athletes_flagged_count == 3
    assert "Alex Vance" in briefing.script_text
    assert "capped at 60 minutes" in briefing.script_text
    assert "prescribe a 300-meter high-speed running primer" in briefing.script_text
    assert "zero lookahead bias" in briefing.script_text


def test_audio_wav_file_generation(tmp_path):
    """Verifies that the synthetic audio generator creates a valid PCM WAV file."""
    synthesizer = MorningAudioBriefingSynthesizer()
    out_file = str(tmp_path / "morning_chime.wav")
    synthesizer.generate_synthetic_audio_wav(out_file, duration_seconds=1.0)
    
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 1000  # Non-empty WAV file


def test_tactical_video_storyboard_and_svg_render():
    """Verifies video storyboard generation and animated SVG frame rendering."""
    video_gen = TacticalVideoGenerator()
    athletes = get_risk_ranking(top_n=3)
    storyboard = video_gen.build_morning_video_storyboard(athletes)

    assert len(storyboard) == 4
    assert storyboard[0].visual_component == "SQUAD_RADAR_HEATMAP"
    assert storyboard[1].visual_component == "PLAYER_CARDS_SHAP_WATERFALL"
    assert storyboard[2].visual_component == "DYNAMIC_TRAJECTORY_CURVES"

    # Test SVG frame generation
    svg = video_gen.render_animated_svg_frame(storyboard[2])
    assert "<svg" in svg
    assert "DYNAMIC_TRAJECTORY_CURVES" in svg
    assert "AthleteIQ 2.0 // Video Breakdown" in svg
