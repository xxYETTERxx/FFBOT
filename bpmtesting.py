import numpy as np
import pandas as pd
import aubio
import time
import json
from datetime import datetime
from pathlib import Path
from bpm_detector import BPMDetector

class BPMTestFramework:
    def __init__(self):
        # Test parameters
        self.hop_sizes = [256, 512, 1024]
        self.aubio_methods = ['default', 'energy', 'complex', 'phase', 'specflux']
        self.samples_per_config = 1000
        
        # Results storage
        self.results_dir = Path('test_results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Metadata tracking
        self.metadata = {
            'test_start': None,
            'test_end': None,
            'total_samples': 0,
            'configurations_tested': []
        }
        
    def test_configuration(self, hop_size, aubio_method):
        """Test a specific configuration of parameters"""
        detector = BPMDetector()
        detector.hop_size = hop_size
        detector.tempo = detector.tempo = aubio.tempo(
            method=aubio_method,
            buf_size=detector.hop_size * 4,
            hop_size=detector.hop_size,
            samplerate=detector.sample_rate
        )
        
        results = []
        start_time = time.time()
        
        with detector.setup_audio_stream():
            for _ in range(self.samples_per_config):
                try:
                    bpm_data = detector.get_bpm()
                    if bpm_data:
                        result = {
                            'timestamp': time.time(),
                            'hop_size': hop_size,
                            'aubio_method': aubio_method,
                            'librosa_bpm': bpm_data['librosa']['bpm'],
                            'aubio_bpm': bpm_data['aubio']['bpm'],
                            'aubio_confidence': bpm_data['aubio']['confidence'],
                            'processing_time': time.time() - start_time
                        }
                        results.append(result)
                        
                except Exception as e:
                    print(f"Error collecting sample: {e}")
                
                time.sleep(0.1)  # Prevent overwhelming the system
                
        return results
    
    def run_tests(self):
        """Run full test suite across all configurations"""
        self.metadata['test_start'] = datetime.now().isoformat()
        all_results = []
        
        for hop_size in self.hop_sizes:
            for method in self.aubio_methods:
                print(f"Testing hop_size={hop_size}, method={method}")
                config_results = self.test_configuration(hop_size, method)
                all_results.extend(config_results)
                
                self.metadata['configurations_tested'].append({
                    'hop_size': hop_size,
                    'aubio_method': method,
                    'samples_collected': len(config_results)
                })
                
        self.metadata['test_end'] = datetime.now().isoformat()
        self.metadata['total_samples'] = len(all_results)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(self.results_dir / f'bpm_test_results_{timestamp}.csv', index=False)
        
        with open(self.results_dir / f'test_metadata_{timestamp}.json', 'w') as f:
            json.dump(self.metadata, f, indent=4)
            
        return results_df
    
    def analyze_results(self, results_df):
        """Analyze test results and generate insights"""
        analysis = {}
        
        # Overall statistics
        analysis['overall'] = {
            'total_samples': len(results_df),
            'avg_processing_time': results_df['processing_time'].mean(),
            'std_processing_time': results_df['processing_time'].std()
        }
        
        # Per-configuration analysis
        for hop_size in self.hop_sizes:
            for method in self.aubio_methods:
                config_mask = (results_df['hop_size'] == hop_size) & (results_df['aubio_method'] == method)
                config_data = results_df[config_mask]
                
                if len(config_data) > 0:
                    analysis[f'hop{hop_size}_{method}'] = {
                        'samples': len(config_data),
                        'avg_aubio_confidence': config_data['aubio_confidence'].mean(),
                        'std_aubio_confidence': config_data['aubio_confidence'].std(),
                        'bpm_correlation': config_data['librosa_bpm'].corr(config_data['aubio_bpm']),
                        'avg_processing_time': config_data['processing_time'].mean()
                    }
        
        # Save analysis
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(self.results_dir / f'analysis_results_{timestamp}.json', 'w') as f:
            json.dump(analysis, f, indent=4)
            
        return analysis

def main():
    print("Starting BPM Detection Test Framework")
    framework = BPMTestFramework()
    
    print("\nRunning tests across all configurations...")
    results = framework.run_tests()
    
    print("\nAnalyzing results...")
    analysis = framework.analyze_results(results)
    
    print("\nTest complete! Results saved to the test_results directory.")
    print(f"Total samples collected: {analysis['overall']['total_samples']}")
    
if __name__ == "__main__":
    main()