import React from 'react';
import {
  Container,
  Typography,
  Box,
  Paper,
  Grid,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip
} from '@mui/material';
import {
  Psychology as PsychologyIcon,
  Security as SecurityIcon,
  Speed as SpeedIcon,
  Visibility as VisibilityIcon,
  Code as CodeIcon,
  Science as ScienceIcon
} from '@mui/icons-material';

const AboutPage: React.FC = () => {
  const technologies = [
    'Xception Neural Network',
    'Convolutional Neural Networks (CNN)',
    'Transfer Learning',
    'Classical Image Processing',
    'Edge Detection Algorithms',
    'Texture Analysis',
    'Compression Artifact Detection'
  ];

  const features = [
    {
      icon: <SecurityIcon sx={{ color: '#1976d2' }} />,
      title: 'Advanced ML Models',
      description: 'Utilizes state-of-the-art pre-trained models like Xception for accurate deepfake detection.'
    },
    {
      icon: <PsychologyIcon sx={{ color: '#1976d2' }} />,
      title: 'Explainable AI',
      description: 'Provides transparent, step-by-step explanations of detection decisions for educational purposes.'
    },
    {
      icon: <VisibilityIcon sx={{ color: '#1976d2' }} />,
      title: 'Visual Analysis',
      description: 'Offers visual representations of detection artifacts and processing steps.'
    },
    {
      icon: <SpeedIcon sx={{ color: '#1976d2' }} />,
      title: 'Real-time Processing',
      description: 'Fast image analysis with progress indicators and immediate feedback.'
    }
  ];

  return (
    <Container maxWidth="lg">
      <Typography variant="h3" component="h1" gutterBottom align="center">
        About IntelliFakeLens
      </Typography>
      
      <Box sx={{ mb: 6 }}>
        <Paper sx={{ p: 4, backgroundColor: '#f8f9fa' }}>
          <Typography variant="h5" gutterBottom>
            Project Overview
          </Typography>
          <Typography variant="body1" paragraph>
            IntelliFakeLens is a cutting-edge web application designed to detect deepfake and manipulated images 
            using advanced machine learning techniques. Our platform combines pre-trained neural networks with 
            classical image processing methods to provide accurate detection results with transparent explanations.
          </Typography>
          <Typography variant="body1" paragraph>
            The primary goal is to contribute to explainable AI in digital forensics by making detection decisions 
            transparent, interpretable, and educational for users. Whether you're a researcher, journalist, or 
            concerned citizen, IntelliFakeLens empowers you to identify potentially manipulated media.
          </Typography>
        </Paper>
      </Box>

      <Grid container spacing={4} sx={{ mb: 6 }}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h5" gutterBottom>
              Key Features
            </Typography>
            <List>
              {features.map((feature, index) => (
                <ListItem key={index} sx={{ px: 0 }}>
                  <ListItemIcon>
                    {feature.icon}
                  </ListItemIcon>
                  <ListItemText
                    primary={feature.title}
                    secondary={feature.description}
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h5" gutterBottom>
              Technologies Used
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 3 }}>
              {technologies.map((tech, index) => (
                <Chip
                  key={index}
                  label={tech}
                  variant="outlined"
                  sx={{ mb: 1 }}
                />
              ))}
            </Box>
            <Typography variant="body2" color="text.secondary">
              Our detection pipeline combines multiple approaches including deep learning models, 
              traditional computer vision techniques, and statistical analysis to ensure robust 
              and reliable results.
            </Typography>
          </Paper>
        </Grid>
      </Grid>

      <Paper sx={{ p: 4, mb: 4 }}>
        <Typography variant="h5" gutterBottom>
          How Detection Works
        </Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} md={3}>
            <Box sx={{ textAlign: 'center' }}>
              <CodeIcon sx={{ fontSize: 40, color: '#1976d2', mb: 1 }} />
              <Typography variant="h6" gutterBottom>
                Preprocessing
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Image normalization, face detection, and feature preparation for analysis.
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={12} md={3}>
            <Box sx={{ textAlign: 'center' }}>
              <ScienceIcon sx={{ fontSize: 40, color: '#1976d2', mb: 1 }} />
              <Typography variant="h6" gutterBottom>
                Feature Extraction
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Deep learning models extract relevant features and patterns from the image.
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={12} md={3}>
            <Box sx={{ textAlign: 'center' }}>
              <PsychologyIcon sx={{ fontSize: 40, color: '#1976d2', mb: 1 }} />
              <Typography variant="h6" gutterBottom>
                Model Inference
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Xception and other models analyze features to make predictions about authenticity.
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={12} md={3}>
            <Box sx={{ textAlign: 'center' }}>
              <VisibilityIcon sx={{ fontSize: 40, color: '#1976d2', mb: 1 }} />
              <Typography variant="h6" gutterBottom>
                Result Analysis
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Confidence scores and visual explanations provide transparent detection results.
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </Paper>

      <Paper sx={{ p: 4, backgroundColor: '#e3f2fd' }}>
        <Typography variant="h5" gutterBottom>
          Important Disclaimer
        </Typography>
        <Typography variant="body1" paragraph>
          While IntelliFakeLens uses advanced machine learning techniques, no detection system is 100% accurate. 
          Results should be considered as part of a broader analysis and verification process. The technology is 
          constantly evolving, and new manipulation techniques may not be detected by current models.
        </Typography>
        <Typography variant="body2" color="text.secondary">
          This tool is designed for educational and research purposes. For critical applications, 
          please consult with digital forensics experts and use multiple verification methods.
        </Typography>
      </Paper>
    </Container>
  );
};

export default AboutPage;