import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Container, AppBar, Toolbar, Typography, Box } from '@mui/material';
import DetectionPage from './components/DetectionPage';
import Navigation from './components/Navigation';

const App: React.FC = () => {
  return (
    <Box sx={{ flexGrow: 1, minHeight: '100vh', backgroundColor: '#f5f5f5' }}>
      <AppBar position="static" elevation={0} sx={{ backgroundColor: '#1976d2' }}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1, fontWeight: 600 }}>
            IntelliFakeLens
          </Typography>
          <Navigation />
        </Toolbar>
      </AppBar>
      
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Routes>
          <Route path="/" element={<DetectionPage />} />
          <Route path="/detect" element={<DetectionPage />} />
        </Routes>
      </Container>
    </Box>
  );
};

export default App;