#!/usr/bin/env python3
"""
Audio-Visual Data Collection Script
Course: Multimodal Interaction - Sapienza University
Author: Diego
Date: 2026-02-01

This script facilitates synchronized audio-video recording for keyword spotting dataset creation.
Features:
- MediaPipe Face Mesh visualization for lip landmarks
- Synchronized audio (44.1kHz) and video (30fps) capture
- Structured dataset organization
- Progress tracking with sample counter
- Visual countdown before recording
"""

import os
import sys

# Suppress Qt threading warnings - these are harmless but verbose
# The warnings occur because OpenCV's Qt backend and MediaPipe both use Qt internally
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false;*.warning=false'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

import cv2
import numpy as np
import mediapipe as mp
import pyaudio
import wave
import threading
import json
from datetime import datetime
from pathlib import Path
import time

# Configure OpenCV to minimize threading conflicts
cv2.setNumThreads(1)  # Single-threaded operation to avoid Qt conflicts


class AVDataCollector:
    """Audio-Visual Data Collector with synchronized capture"""
    
    def __init__(self, output_dir="dataset", keywords=None):
        """
        Initialize the data collector.
        
        Args:
            output_dir: Root directory for dataset storage
            keywords: List of keywords to record (default: 10 standard commands)
        """
        # Italian keywords as per specification
        if keywords is None:
            self.keywords = [
                "Avvia", "Stop", "Sopra", "Sotto", "Sinistra", 
                "Destra", "Apri", "Chiudi", "Sì", "No"
            ]
        else:
            self.keywords = keywords
        
        # Directory structure
        self.output_dir = Path(output_dir)
        self.create_directory_structure()
        
        # Video capture settings
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # 30fps as per spec
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # MediaPipe Face Mesh for lip visualization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Lip landmarks indices (MediaPipe Face Mesh specific indices)
        # Outer lips
        self.LIP_LANDMARKS = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375,  # Upper outer lip
            291, 409, 270, 269, 267, 0, 37, 39, 40, 185     # Lower outer lip
        ]
        
        # Inner lips
        self.INNER_LIP_LANDMARKS = [
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324,   # Upper inner lip
            308, 415, 310, 311, 312, 13, 82, 81, 80, 191    # Lower inner lip
        ]
        
        # Audio settings (44.1kHz as per spec)
        self.AUDIO_FORMAT = pyaudio.paInt16
        self.CHANNELS = 1  # Mono
        self.RATE = 44100  # 44.1kHz
        self.CHUNK = 1024
        self.RECORD_SECONDS = 2  # Duration per sample
        
        self.audio = pyaudio.PyAudio()
        
        # Recording state
        self.is_recording = False
        self.audio_frames = []
        
        # Load or initialize progress tracking
        self.progress_file = self.output_dir / "progress.json"
        self.load_progress()
        
        # UI state
        self.current_keyword_idx = 0
        self.countdown_active = False
        
    def create_directory_structure(self):
        """Create organized directory structure for dataset"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for keyword in self.keywords:
            # Video directory
            (self.output_dir / keyword / "video").mkdir(parents=True, exist_ok=True)
            # Audio directory
            (self.output_dir / keyword / "audio").mkdir(parents=True, exist_ok=True)
            # Landmarks directory (for extracted features)
            # (self.output_dir / keyword / "landmarks").mkdir(parents=True, exist_ok=True)
    
    def load_progress(self):
        """Load recording progress from JSON file"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
        else:
            self.progress = {keyword: 0 for keyword in self.keywords}
            self.save_progress()
    
    def save_progress(self):
        """Save recording progress to JSON file"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, indent=2, fp=f)
    
    def get_next_sample_number(self, keyword):
        """Get the next sample number for a keyword"""
        return self.progress[keyword] + 1
    
    def audio_recording_thread(self, filename, stream, start_event):
        """Thread function for audio recording - Now accepts pre-opened stream"""
        # Wait for sync signal from main thread
        start_event.wait()
        
        print(f"🎤 Recording audio to {filename}")
        self.audio_frames = []
        
        # Use the passed stream
        start_time = time.time()
        while self.is_recording and (time.time() - start_time) < self.RECORD_SECONDS:
            data = stream.read(self.CHUNK, exception_on_overflow=False)
            self.audio_frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        # Save audio file
        wf = wave.open(str(filename), 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(self.AUDIO_FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.audio_frames))
        wf.close()
        
        print(f"✅ Audio saved: {filename}")
    
    def draw_lip_landmarks(self, frame, face_landmarks):
        """Draw lip landmarks on frame for visual feedback"""
        h, w, _ = frame.shape
        
        # Draw outer lips
        for idx in self.LIP_LANDMARKS:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        # Draw inner lips
        for idx in self.INNER_LIP_LANDMARKS:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        
        return frame
    
    def extract_lip_landmarks(self, face_landmarks):
        """Extract lip landmark coordinates for saving"""
        landmarks_data = {
            'outer_lips': [],
            'inner_lips': []
        }
        
        for idx in self.LIP_LANDMARKS:
            landmark = face_landmarks.landmark[idx]
            landmarks_data['outer_lips'].append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            })
        
        for idx in self.INNER_LIP_LANDMARKS:
            landmark = face_landmarks.landmark[idx]
            landmarks_data['inner_lips'].append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            })
        
        return landmarks_data
    
    def align_audio_video_duration(self, video_frames_count, audio_path, fps=30.0):
        """
        Trim audio to match exact video duration for frame-accurate synchronization.
        
        Args:
            video_frames_count: Number of video frames captured
            audio_path: Path to the audio file
            fps: Video frame rate
        
        Returns:
            True if alignment successful
        """
        try:
            # Calculate exact video duration
            video_duration = video_frames_count / fps
            target_samples = int(video_duration * self.RATE)
            
            # Read original audio file
            wf = wave.open(str(audio_path), 'rb')
            audio_data = wf.readframes(wf.getnframes())
            original_frames = wf.getnframes()
            params = wf.getparams()
            wf.close()
            
            # Convert to numpy for easier manipulation
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate original duration
            original_duration = original_frames / self.RATE
            
            # Trim or pad to exact target length
            if len(audio_array) > target_samples:
                # Trim excess
                audio_array = audio_array[:target_samples]
                print(f"  🔧 Trimmed audio: {original_duration:.3f}s → {video_duration:.3f}s ({original_frames - target_samples} samples)")
            elif len(audio_array) < target_samples:
                # Pad with silence (rare case)
                padding = np.zeros(target_samples - len(audio_array), dtype=np.int16)
                audio_array = np.concatenate([audio_array, padding])
                print(f"  🔧 Padded audio: {original_duration:.3f}s → {video_duration:.3f}s (+{target_samples - original_frames} samples)")
            else:
                print(f"  ✅ Audio already aligned: {video_duration:.3f}s")
            
            # Write aligned audio back to file
            wf = wave.open(str(audio_path), 'wb')
            wf.setparams(params)
            wf.writeframes(audio_array.tobytes())
            wf.close()
            
            return True
            
        except Exception as e:
            print(f"  ⚠️  Audio alignment failed: {e}")
            return False
    
    def record_sample(self, keyword):
        """Record a single audio-video sample"""
        sample_num = self.get_next_sample_number(keyword)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # File paths
        video_path = self.output_dir / keyword / "video" / f"{keyword}_{sample_num:03d}_{timestamp}.avi"
        audio_path = self.output_dir / keyword / "audio" / f"{keyword}_{sample_num:03d}_{timestamp}.wav"
        # landmarks_path = self.output_dir / keyword / "landmarks" / f"{keyword}_{sample_num:03d}_{timestamp}.json"
        
        # Video writer setup (30fps)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 30.0
        frame_size = (640, 480)
        out = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)
        
        # --- AUDIO SETUP (Pre-initialization for Sync) ---
        stream = self.audio.open(
            format=self.AUDIO_FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        # Sync event
        start_event = threading.Event()
        
        # Start audio recording thread
        self.is_recording = True
        audio_thread = threading.Thread(
            target=self.audio_recording_thread, 
            args=(audio_path, stream, start_event), 
            daemon=True
        )
        audio_thread.start()
        
        # Video recording preparation
        print(f"🎥 Recording video to {video_path}")
        
        # RAM Buffer for frames (Capture First, Process Later)
        frames_buffer = []
        
        # --- CRITICAL SYNC POINT ---
        print("🔄 Sincronizzazione: svuotamento buffer...")
        
        # 1. Flush Audio Buffer
        # Read all available data from the stream to flush the buffer
        if stream.get_read_available() > 0:
            bytes_to_read = stream.get_read_available()
            stream.read(bytes_to_read, exception_on_overflow=False)
            
        # 2. Flush Video Buffer
        # Read a few frames to ensure we have the latest available frame
        # Webcam buffer typically has 1-2 frames of latency
        for _ in range(5):
            self.cap.grab()
            
        # Signal the start of recording
        start_event.set()
        start_time = time.time()
        
        # --- PHASE 1: HIGH SPEED CAPTURE ---
        while (time.time() - start_time) < self.RECORD_SECONDS:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Store raw frame in RAM
            frames_buffer.append(frame)
            
            # Draw simple UI on a copy for display only (don't save UI overlay)
            display_frame = frame.copy()
            cv2.circle(display_frame, (20, 20), 10, (0, 0, 255), -1)
            cv2.putText(display_frame, "REC", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2)
            cv2.imshow('Data Collection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Stop recording
        end_time = time.time()
        self.is_recording = False
        audio_thread.join(timeout=1.0)
        
        # Calculate actual FPS
        actual_duration = end_time - start_time
        if actual_duration > 0 and len(frames_buffer) > 0:
            actual_fps = len(frames_buffer) / actual_duration
            print(f"📊 Actual FPS: {actual_fps:.2f} (Captured {len(frames_buffer)} frames in {actual_duration:.2f}s)")
        else:
            actual_fps = 30.0
            print("⚠️ Could not calculate actual FPS, defaulting to 30.0")

        # Update VideoWriter with actual FPS
        out = cv2.VideoWriter(str(video_path), fourcc, actual_fps, frame_size)
        
        # --- PHASE 2: POST-PROCESSING (MediaPipe & Save) ---
        print(f"⚡ Processing {len(frames_buffer)} frames...")
        all_landmarks = []
        
        for i, frame in enumerate(frames_buffer):
            # Process with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            # Draw landmarks on a copy, save the clean frame!
            display_frame = frame.copy()
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw for visual feedback ONLY on the display copy
                    display_frame = self.draw_lip_landmarks(display_frame, face_landmarks)
                    
                    # Extract landmarks
                    landmarks_data = self.extract_lip_landmarks(face_landmarks)
                    all_landmarks.append(landmarks_data)
            
            # Write the CLEAN, UNMODIFIED frame to file
            out.write(frame)
            
            # Optional: Show progress frame (useful if we want to see post-processing live)
            # cv2.imshow('Processing', display_frame)
            # cv2.waitKey(1)
            
            # Show progress every 10 frames
            if i % 10 == 0:
                print(f"  Processed {i}/{len(frames_buffer)}...", end='\r')
                
        out.release()
        
        # Save landmarks
        # with open(landmarks_path, 'w') as f:
        #     json.dump(all_landmarks, indent=2, fp=f)
        
        print(f"\n✅ Video saved: {video_path}")
        # print(f"✅ Landmarks saved: {landmarks_path}")
        
        # Align audio to exact video duration for perfect sync
        print(f"🔄 Aligning audio-video synchronization...")
        self.align_audio_video_duration(len(frames_buffer), audio_path, actual_fps)
        
        # Update progress
        self.progress[keyword] += 1
        self.save_progress()
        
        return True
    
    def draw_ui(self, frame):
        """Draw UI overlay on frame"""
        h, w, _ = frame.shape
        
        # Semi-transparent overlay
        overlay = frame.copy()
        
        # Top bar with current keyword
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Keyword corrente
        current_keyword = self.keywords[self.current_keyword_idx]
        cv2.putText(frame, f"Parola: {current_keyword}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Progresso
        progress_text = f"Campioni: {self.progress[current_keyword]}/20"
        cv2.putText(frame, progress_text, (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Bottom instruction bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-80), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Istruzioni
        cv2.putText(frame, "SPAZIO: Registra | N: Parola Successiva | P: Parola Precedente | Q: Esci",
                   (20, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Parla chiaramente e guarda la camera!",
                   (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        
        return frame
    
    def show_countdown(self, frame):
        """Show countdown before recording"""
        h, w, _ = frame.shape
        for i in range(3, 0, -1):
            ret, countdown_frame = self.cap.read()
            if not ret:
                continue
            
            # Process with MediaPipe for visual feedback
            frame_rgb = cv2.cvtColor(countdown_frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    countdown_frame = self.draw_lip_landmarks(countdown_frame, face_landmarks)
            
            # Draw countdown number
            cv2.putText(countdown_frame, str(i), (w//2 - 50, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)
            
            # Draw UI
            countdown_frame = self.draw_ui(countdown_frame)
            
            cv2.imshow('Data Collection', countdown_frame)
            cv2.waitKey(1000)
    
    def run(self):
        """Loop principale di raccolta dati"""
        print("=" * 60)
        print("🎬 Sistema di Raccolta Dati Audio-Video")
        print("=" * 60)
        print("\nParole da registrare:")
        for i, keyword in enumerate(self.keywords, 1):
            print(f"  {i}. {keyword} ({self.progress[keyword]}/20 campioni)")
        print("\nControlli:")
        print("  SPAZIO - Avvia registrazione parola corrente")
        print("  N      - Parola successiva")
        print("  P      - Parola precedente")
        print("  Q      - Esci")
        print("=" * 60)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Error: Cannot read from camera")
                break
            
            # Process frame with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            # Draw lip landmarks if face detected
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    frame = self.draw_lip_landmarks(frame, face_landmarks)
            else:
                # Avviso se nessun volto rilevato
                cv2.putText(frame, "Nessun volto rilevato!", (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Draw UI
            frame = self.draw_ui(frame)
            
            # Display
            cv2.imshow('Data Collection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Uscita dalla raccolta dati...")
                break
            
            elif key == ord(' '):
                current_keyword = self.keywords[self.current_keyword_idx]
                
                if self.progress[current_keyword] >= 20:
                    print(f"⚠️  Già raccolti 20 campioni per '{current_keyword}'")
                    continue
                
                print(f"\n📹 Preparazione registrazione di '{current_keyword}'...")
                self.show_countdown(frame)
                self.record_sample(current_keyword)
                
                print(f"✨ Campione {self.progress[current_keyword]}/20 completato!")
                
                # Avanzamento automatico se raggiunti 20 campioni
                if self.progress[current_keyword] >= 20:
                    print(f"🎉 Completati tutti i campioni per '{current_keyword}'!")
                    if self.current_keyword_idx < len(self.keywords) - 1:
                        self.current_keyword_idx += 1
                        print(f"➡️  Passaggio alla parola successiva: '{self.keywords[self.current_keyword_idx]}'")
            
            elif key == ord('n'):
                if self.current_keyword_idx < len(self.keywords) - 1:
                    self.current_keyword_idx += 1
                    print(f"\n➡️  Cambiato a: {self.keywords[self.current_keyword_idx]}")
                else:
                    print("\n⚠️  Già all'ultima parola")
            
            elif key == ord('p'):
                if self.current_keyword_idx > 0:
                    self.current_keyword_idx -= 1
                    print(f"\n⬅️  Cambiato a: {self.keywords[self.current_keyword_idx]}")
                else:
                    print("\n⚠️  Già alla prima parola")
        
        # Cleanup
        self.cleanup()
        self.print_summary()
    
    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.audio.terminate()
    
    def print_summary(self):
        """Stampa riepilogo raccolta"""
        print("\n" + "=" * 60)
        print("📊 RIEPILOGO RACCOLTA DATI")
        print("=" * 60)
        
        total_samples = 0
        for keyword in self.keywords:
            count = self.progress[keyword]
            total_samples += count
            status = "✅" if count >= 20 else "⚠️ "
            print(f"  {status} {keyword:10s}: {count:2d}/20 campioni")
        
        print("-" * 60)
        print(f"  Totale: {total_samples}/{len(self.keywords) * 20} campioni")
        
        completion = (total_samples / (len(self.keywords) * 20)) * 100
        print(f"  Completamento: {completion:.1f}%")
        print("=" * 60)
        
        if total_samples == len(self.keywords) * 20:
            print("🎉 Congratulazioni! Raccolta dataset completata!")
        
        print(f"\n📁 Dataset salvato in: {self.output_dir.absolute()}")


def main():
    """Punto di ingresso principale"""
    # Create collector instance
    collector = AVDataCollector(output_dir="dataset")
    
    # Start collection loop
    try:
        collector.run()
    except KeyboardInterrupt:
        print("\n⚠️  Interrotto dall'utente")
        collector.cleanup()
        collector.print_summary()
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        collector.cleanup()


if __name__ == "__main__":
    main()
