import os
import shutil
import pretty_midi

def is_silent_or_invalid(midi_path, min_duration=1.0, max_silence_duration=3.0):
    """
    Returns True if the MIDI is:
    - unreadable
    - very short
    - completely silent
    - contains a silence gap longer than `max_silence_duration` in any instrument
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        duration = midi.get_end_time()
        if duration < min_duration:
            return True

        has_any_notes = False
        for instr in midi.instruments:
            notes = sorted(instr.notes, key=lambda n: n.start)
            if not notes:
                continue
            has_any_notes = True
            # Check silence gaps between consecutive notes
            for i in range(1, len(notes)):
                gap = notes[i].start - notes[i - 1].end
                if gap > max_silence_duration:
                    return True

        return not has_any_notes  # return True if all instruments are empty

    except Exception as e:
        print(f"⚠️ Error reading {midi_path}: {e}")
        return True

# Base directories
base_dir = os.path.dirname(__file__)
fixed_dir = os.path.join(base_dir, "fixed_midi")
bad_dir = os.path.join(base_dir, "bad_midis")

# Parameters
min_duration = 1.0              # seconds
max_silence_duration = 3.0      # seconds

moved_count = 0
for root, _, files in os.walk(fixed_dir):
    for f in files:
        if f.lower().endswith(('.mid', '.midi')):
            full_path = os.path.join(root, f)
            if is_silent_or_invalid(full_path, min_duration, max_silence_duration):
                rel_path = os.path.relpath(full_path, fixed_dir)
                dest_path = os.path.join(bad_dir, rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.move(full_path, dest_path)
                print(f"🚫 Moved to bad_midis: {rel_path}")
                moved_count += 1

print(f"\n✅ 检查完成，共移动 {moved_count} 个无效或有长时间静音的 MIDI 文件到 'bad_midis/'")