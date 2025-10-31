import React, { useState, useCallback } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Alert,
  Button,
  Stepper,
  Step,
  StepLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Dialog,
  DialogContent,
  DialogTitle,
  IconButton
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import {
  CloudUpload as CloudUploadIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Visibility as VisibilityIcon,
  ExpandMore as ExpandMoreIcon,
  ZoomIn as ZoomInIcon,
  Close as CloseIcon
} from '@mui/icons-material';
import axios from 'axios';

interface DetectionResult {
  prediction: string;
  confidence: number;
  model_used: string;
  analysis_steps: AnalysisStep[];
  raw_predictions?: Array<{
    label: string;
    score: number;
  }>;
}

interface AnalysisStep {
  step_number: number;
  step_name: string;
  finding: string;
  interpretation: string;
  visualization?: string;
}

const DetectionPage: React.FC = () => {
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);
  const [activeStep, setActiveStep] = useState(-1);
  const [expandedStep, setExpandedStep] = useState<string | false>(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [dialogImage, setDialogImage] = useState<string | null>(null);
  const [dialogTitle, setDialogTitle] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setUploadedImage(file);
      setImagePreview(URL.createObjectURL(file));
      setDetectionResult(null);
      setError(null); // Clear any previous errors
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.bmp', '.webp']
    },
    multiple: false
  });

  const handleDetection = async () => {
    if (!uploadedImage) return;

    setIsProcessing(true);
    setActiveStep(0);
    setError(null);
    setDetectionResult(null);

    try {
      const formData = new FormData();
      formData.append('image', uploadedImage);

      // Simulate processing steps for demo
      const steps = [
        'Preprocessing image',
        'Feature extraction',
        'Model inference',
        'Analysis complete'
      ];

      for (let i = 0; i < steps.length; i++) {
        setActiveStep(i);
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      const response = await axios.post('http://localhost:5000/api/detect', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setDetectionResult(response.data);
      setActiveStep(steps.length);
    } catch (err: any) {
      console.error('Detection failed:', err);
      let errorMessage = 'Detection failed. ';
      
      if (err.code === 'ECONNREFUSED' || err.message?.includes('ECONNREFUSED')) {
        errorMessage += 'Cannot connect to backend server. Please make sure the backend is running on port 5000.';
      } else if (err.response?.status === 500) {
        errorMessage += `Server error: ${err.response.data?.error || 'Internal server error'}`;
      } else if (err.response?.status === 400) {
        errorMessage += `Bad request: ${err.response.data?.error || 'Invalid image or request'}`;
      } else if (err.response?.data?.error) {
        errorMessage += err.response.data.error;
      } else {
        errorMessage += err.message || 'Unknown error occurred';
      }
      
      setError(errorMessage);
      setActiveStep(-1);
    } finally {
      setIsProcessing(false);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return '#4caf50';
    if (confidence >= 0.6) return '#ff9800';
    return '#f44336';
  };

  const handleExpandStep = (stepName: string) => {
    setExpandedStep(expandedStep === stepName ? false : stepName);
  };

  const openImageDialog = (imageData: string, title: string) => {
    setDialogImage(imageData);
    setDialogTitle(title);
    setDialogOpen(true);
  };

  const closeImageDialog = () => {
    setDialogOpen(false);
    setDialogImage(null);
    setDialogTitle('');
  };

  const steps = [
    'Preprocessing image',
    'Feature extraction', 
    'Model inference',
    'Analysis complete'
  ];

  return (
    <Box>
      <Typography variant="h3" component="h1" gutterBottom align="center">
        Deepfake Detection
      </Typography>
      <Typography variant="body1" align="center" color="text.secondary" sx={{ mb: 4 }}>
        Upload an image to analyze it for potential deepfake or manipulation artifacts
      </Typography>

      <Grid container spacing={4}>
        {/* Upload Section */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              Upload Image
            </Typography>
            
            <Box
              {...getRootProps()}
              className={`upload-area ${isDragActive ? 'drag-over' : ''}`}
              sx={{ mb: 2 }}
            >
              <input {...getInputProps()} />
              <CloudUploadIcon sx={{ fontSize: 48, color: '#ccc', mb: 2 }} />
              <Typography variant="body1" color="text.secondary">
                {isDragActive
                  ? 'Drop the image here...'
                  : 'Drag & drop an image here, or click to select'}
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                Supports: JPG, PNG, GIF, BMP, WebP
              </Typography>
            </Box>

            {imagePreview && (
              <Box sx={{ mt: 2 }}>
                <img
                  src={imagePreview}
                  alt="Preview"
                  style={{
                    width: '100%',
                    maxHeight: '300px',
                    objectFit: 'contain',
                    borderRadius: '8px'
                  }}
                />
                <Button
                  variant="contained"
                  onClick={handleDetection}
                  disabled={isProcessing}
                  fullWidth
                  sx={{ mt: 2 }}
                >
                  {isProcessing ? 'Analyzing...' : 'Analyze Image'}
                </Button>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Results Section */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, minHeight: '400px' }}>
            <Typography variant="h5" gutterBottom>
              Analysis Results
            </Typography>

            {isProcessing && (
              <Box>
                <Stepper activeStep={activeStep} orientation="vertical">
                  {steps.map((label) => (
                    <Step key={label}>
                      <StepLabel>{label}</StepLabel>
                    </Step>
                  ))}
                </Stepper>
              </Box>
            )}

            {error && !isProcessing && (
              <Alert severity="error" sx={{ mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Detection Failed
                </Typography>
                <Typography variant="body2">
                  {error}
                </Typography>
                <Button 
                  variant="outlined" 
                  size="small" 
                  sx={{ mt: 2 }}
                  onClick={() => setError(null)}
                >
                  Dismiss
                </Button>
              </Alert>
            )}

            {detectionResult && !isProcessing && (
              <Box className="animate-fade-in">
                <Alert
                  severity={detectionResult.prediction.includes('human') ? 'success' : 'error'}
                  icon={detectionResult.prediction.includes('human') ? <CheckCircleIcon /> : <ErrorIcon />}
                  sx={{ mb: 3 }}
                >
                  <Typography variant="h6">
                    Prediction: {detectionResult.prediction}
                  </Typography>
                  <Typography variant="body2">
                    Confidence: {(detectionResult.confidence * 100).toFixed(1)}%
                  </Typography>
                </Alert>

                <Box sx={{ mb: 3 }}>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Overall Confidence
                  </Typography>
                  <Box className="confidence-meter">
                    <Box
                      className="confidence-fill"
                      sx={{
                        width: `${detectionResult.confidence * 100}%`,
                        backgroundColor: getConfidenceColor(detectionResult.confidence)
                      }}
                    />
                  </Box>
                  <Typography variant="caption" color="text.secondary">
                    Model: {detectionResult.model_used}
                  </Typography>
                </Box>

                {detectionResult.raw_predictions && detectionResult.raw_predictions.length > 0 && (
                  <Paper sx={{ p: 2, mb: 3, backgroundColor: '#f5f5f5' }}>
                    <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 'bold' }}>
                      Raw Model Predictions:
                    </Typography>
                    {detectionResult.raw_predictions.slice(0, 5).map((pred, idx) => (
                      <Box key={idx} sx={{ mb: 1 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                            {pred.label}
                          </Typography>
                          <Chip 
                            label={`${(pred.score * 100).toFixed(2)}%`}
                            size="small"
                            color={idx === 0 ? 'primary' : 'default'}
                            sx={{ minWidth: '80px' }}
                          />
                        </Box>
                        <Box sx={{ 
                          height: '4px', 
                          backgroundColor: '#e0e0e0', 
                          borderRadius: '2px',
                          mt: 0.5,
                          overflow: 'hidden'
                        }}>
                          <Box sx={{
                            width: `${pred.score * 100}%`,
                            height: '100%',
                            backgroundColor: idx === 0 ? '#1976d2' : '#9e9e9e',
                            transition: 'width 0.3s ease'
                          }} />
                        </Box>
                      </Box>
                    ))}
                  </Paper>
                )}

                <Typography variant="h6" gutterBottom>
                  Analysis Steps
                </Typography>
                <Box className="detection-steps">
                  {detectionResult.analysis_steps.map((step, index) => {
                    const isNeuralLayer = step.step_name.includes('Neural');
                    return (
                    <Accordion 
                      key={index} 
                      expanded={expandedStep === step.step_name}
                      onChange={() => handleExpandStep(step.step_name)}
                      sx={{ 
                        mb: 2,
                        ...(isNeuralLayer && {
                          backgroundColor: '#f3e5f5',
                          border: '1px solid #9c27b0'
                        })
                      }}
                    >
                      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Box sx={{ width: '100%', display: 'flex', alignItems: 'center', gap: 2 }}>
                          <Chip 
                            label={`Step ${step.step_number}`}
                            color="primary"
                            size="small"
                            sx={{ mr: 1 }}
                          />
                          <Typography variant="subtitle1" sx={{ flexGrow: 1 }}>
                            {step.step_name}
                            {isNeuralLayer && (
                              <Chip 
                                label="Neural Network"
                                size="small"
                                variant="outlined"
                                color="secondary"
                                sx={{ ml: 1, fontSize: '0.7rem' }}
                              />
                            )}
                          </Typography>
                        </Box>
                      </AccordionSummary>
                      <AccordionDetails>
                        <Box>
                          <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 'bold' }}>
                            Finding:
                          </Typography>
                          <Typography variant="body2" sx={{ mb: 2 }}>
                            {step.finding}
                          </Typography>
                          
                          <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 'bold' }}>
                            Interpretation:
                          </Typography>
                          <Typography variant="body2" sx={{ mb: 2 }}>
                            {step.interpretation}
                          </Typography>

                          {step.visualization && (
                            <Box sx={{ mt: 2 }}>
                              <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 'bold' }}>
                                Visualization:
                              </Typography>
                              <Paper 
                                elevation={1} 
                                sx={{ 
                                  p: 1, 
                                  textAlign: 'center',
                                  cursor: 'pointer',
                                  '&:hover': { elevation: 3 }
                                }}
                                onClick={() => openImageDialog(
                                  step.visualization && step.visualization.startsWith('data:')
                                    ? step.visualization
                                    : `data:image/png;base64,${step.visualization}`,
                                  `${step.step_name} - Detailed View`
                                )}
                              >
                                <img
                                  src={step.visualization && step.visualization.startsWith('data:') ? step.visualization : `data:image/png;base64,${step.visualization}`}
                                  alt={`${step.step_name} visualization`}
                                  style={{
                                    maxWidth: '100%',
                                    maxHeight: '200px',
                                    objectFit: 'contain'
                                  }}
                                />
                                <Typography variant="caption" display="block" sx={{ mt: 1, color: 'primary.main' }}>
                                  <ZoomInIcon sx={{ fontSize: 14, mr: 0.5 }} />
                                  Click to enlarge
                                </Typography>
                              </Paper>
                              
                              {step.interpretation && (
                                <Alert severity={isNeuralLayer ? "info" : "info"} sx={{ mt: 2 }}>
                                  <Typography variant="body2">
                                    <strong>Interpretation:</strong> {step.interpretation}
                                  </Typography>
                                </Alert>
                              )}
                            </Box>
                          )}
                        </Box>
                      </AccordionDetails>
                    </Accordion>
                    );
                  })}
                </Box>
              </Box>
            )}

            {!uploadedImage && !isProcessing && !error && (
              <Box sx={{ textAlign: 'center', color: 'text.secondary', mt: 4 }}>
                <VisibilityIcon sx={{ fontSize: 64, opacity: 0.3 }} />
                <Typography variant="body1" sx={{ mt: 2 }}>
                  Upload an image to see analysis results
                </Typography>
              </Box>
            )}

            {uploadedImage && !isProcessing && !detectionResult && !error && (
              <Box sx={{ textAlign: 'center', color: 'text.secondary', mt: 4 }}>
                <Button
                  variant="contained"
                  onClick={handleDetection}
                  sx={{ mb: 2 }}
                >
                  Start Analysis
                </Button>
                <Typography variant="body2">
                  Click to analyze the uploaded image for deepfake detection
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>

      {/* Image Dialog for enlarged view */}
      <Dialog 
        open={dialogOpen} 
        onClose={closeImageDialog}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="h6">{dialogTitle}</Typography>
            <IconButton onClick={closeImageDialog}>
              <CloseIcon />
            </IconButton>
          </Box>
        </DialogTitle>
        <DialogContent>
          {dialogImage && (
            <Box sx={{ textAlign: 'center' }}>
              <img
                src={dialogImage}
                alt="Enlarged visualization"
                style={{
                  maxWidth: '100%',
                  maxHeight: '80vh',
                  objectFit: 'contain'
                }}
              />
            </Box>
          )}
        </DialogContent>
      </Dialog>
    </Box>
  );
};

export default DetectionPage;