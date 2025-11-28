
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Mock modules before importing main_pipeline
sys.modules['sounddevice'] = MagicMock()
sys.modules['cv2'] = MagicMock()

# Mock cv2 attributes
sys.modules['cv2'].VideoCapture = MagicMock()
sys.modules['cv2'].imshow = MagicMock()
sys.modules['cv2'].waitKey = MagicMock(return_value=ord('q')) # Return 'q' to exit loop immediately
sys.modules['cv2'].cvtColor = MagicMock(return_value=np.zeros((100, 100, 3), dtype=np.uint8))
sys.modules['cv2'].COLOR_BGR2GRAY = 6
sys.modules['cv2'].rectangle = MagicMock()
sys.modules['cv2'].putText = MagicMock()
sys.modules['cv2'].FONT_HERSHEY_SIMPLEX = 0
sys.modules['cv2'].resize = MagicMock(return_value=np.zeros((100, 100, 3), dtype=np.uint8))
sys.modules['cv2'].flip = MagicMock(return_value=np.zeros((100, 100, 3), dtype=np.uint8))
sys.modules['cv2'].absdiff = MagicMock(return_value=np.zeros((100, 100), dtype=np.uint8))
sys.modules['cv2'].boundingRect = MagicMock(return_value=(0, 0, 10, 10))

# Mock faster_whisper
sys.modules['faster_whisper'] = MagicMock()
mock_whisper_model = MagicMock()
sys.modules['faster_whisper'].WhisperModel = MagicMock(return_value=mock_whisper_model)

# Mock transcribe return value
# faster-whisper returns (segments, info)
# segments is a generator or list of Segment objects
mock_segment = MagicMock()
mock_segment.text = "테스트 텍스트"
mock_whisper_model.transcribe.return_value = ([mock_segment], MagicMock())

# Add src paths
current_dir = os.getcwd()
src_path = os.path.join(current_dir, 'src')
video_path = os.path.join(src_path, 'video')
sys.path.insert(0, video_path)
sys.path.insert(1, src_path)

# Now import main_pipeline
import main_pipeline

class TestMainPipeline(unittest.TestCase):
    @patch('main_pipeline.VideoProcessor')
    @patch('main_pipeline.sd.InputStream')
    def test_run_realtime_pipeline(self, mock_input_stream, mock_video_processor):
        # Setup mocks
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        
        mock_processor_instance = mock_video_processor.return_value
        mock_processor_instance.cap = mock_cap
        mock_processor_instance.process_frame.return_value = {
            "frame_id": 0,
            "timestamp": 0.0,
            "roi": {"x": 0, "y": 0, "w": 10, "h": 10},
            "is_speaking": True,
            "confidence": 0.9,
            "person_detected": True
        }

        # Run the pipeline
        try:
            main_pipeline.run_realtime_pipeline()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fail(f"run_realtime_pipeline raised Exception: {e}")

if __name__ == '__main__':
    unittest.main()
