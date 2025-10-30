import React from 'react';
import { 
  Box, 
  Typography, 
  Button, 
  Container, 
  Grid, 
  Paper,
  Card,
  CardContent
} from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';
import { 
  Security as SecurityIcon, 
  Visibility as VisibilityIcon,
  Speed as SpeedIcon,
  Psychology as PsychologyIcon
} from '@mui/icons-material';

const HomePage: React.FC = () => {
  const features = [
    {
      icon: <SecurityIcon sx={{ fontSize: 40, color: '#1976d2' }} />,
      title: 'Advanced Detection',
      description: 'Uses state-of-the-art ML models like Xception to detect deepfakes and manipulated images with high accuracy.'
    },
    {
      icon: <VisibilityIcon sx={{ fontSize: 40, color: '#1976d2' }} />,
      title: 'Visual Explanations',
      description: 'Provides step-by-step visual explanations of how detection decisions are made for transparency.'
    },
    {
      icon: <SpeedIcon sx={{ fontSize: 40, color: '#1976d2' }} />,
      title: 'Real-time Processing',
      description: 'Fast image analysis with real-time progress indicators and immediate results.'
    },
    {
      icon: <PsychologyIcon sx={{ fontSize: 40, color: '#1976d2' }} />,
      title: 'Explainable AI',
      description: 'Makes AI decisions transparent and interpretable for educational and forensic purposes.'
    }
  ];

  return (
    <Container maxWidth="lg">
      {/* Hero Section */}
      <Box 
        sx={{ 
          textAlign: 'center', 
          py: 8,
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          borderRadius: 3,
          color: 'white',
          mb: 6
        }}
      >
        <Typography variant="h1" component="h1" gutterBottom>
          IntelliFakeLens
        </Typography>
        <Typography variant="h5" component="h2" sx={{ mb: 4, opacity: 0.9 }}>
          Advanced Deepfake Detection with Explainable AI
        </Typography>
        <Typography variant="body1" sx={{ mb: 4, maxWidth: 600, mx: 'auto', opacity: 0.8 }}>
          Upload an image and let our machine learning models analyze it for signs of manipulation, 
          deepfake artifacts, and digital tampering with clear explanations of the detection process.
        </Typography>
        <Button
          component={RouterLink}
          to="/detect"
          variant="contained"
          size="large"
          sx={{ 
            backgroundColor: 'white',
            color: '#1976d2',
            '&:hover': { backgroundColor: '#f5f5f5' },
            textTransform: 'none',
            px: 4,
            py: 1.5
          }}
        >
          Start Detection
        </Button>
      </Box>

      {/* Features Section */}
      <Typography variant="h3" component="h2" align="center" gutterBottom sx={{ mb: 4 }}>
        Key Features
      </Typography>
      
      <Grid container spacing={4} sx={{ mb: 6 }}>
        {features.map((feature, index) => (
          <Grid item xs={12} md={6} key={index}>
            <Card 
              sx={{ 
                height: '100%', 
                display: 'flex', 
                flexDirection: 'column',
                '&:hover': { transform: 'translateY(-4px)' },
                transition: 'transform 0.3s ease'
              }}
            >
              <CardContent sx={{ flexGrow: 1, textAlign: 'center', pt: 3 }}>
                <Box sx={{ mb: 2 }}>
                  {feature.icon}
                </Box>
                <Typography variant="h5" component="h3" gutterBottom>
                  {feature.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {feature.description}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* How it Works Section */}
      <Paper sx={{ p: 4, backgroundColor: '#f8f9fa', borderRadius: 3 }}>
        <Typography variant="h4" component="h2" align="center" gutterBottom>
          How It Works
        </Typography>
        <Grid container spacing={3} sx={{ mt: 2 }}>
          <Grid item xs={12} md={4}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h6" sx={{ color: '#1976d2', mb: 1 }}>
                1. Upload Image
              </Typography>
              <Typography variant="body2">
                Simply drag and drop or select an image you want to analyze for potential manipulation.
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={12} md={4}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h6" sx={{ color: '#1976d2', mb: 1 }}>
                2. AI Analysis
              </Typography>
              <Typography variant="body2">
                Our pre-trained models analyze the image for deepfake artifacts and manipulation signs.
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={12} md={4}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h6" sx={{ color: '#1976d2', mb: 1 }}>
                3. Detailed Results
              </Typography>
              <Typography variant="body2">
                Get comprehensive results with visual explanations and confidence scores.
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </Paper>
    </Container>
  );
};

export default HomePage;