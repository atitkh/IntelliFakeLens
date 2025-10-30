import React from 'react';
import { Button, Box } from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';

const Navigation: React.FC = () => {
  return (
    <Box sx={{ display: 'flex', gap: 2 }}>
      <Button 
        color="inherit" 
        component={RouterLink} 
        to="/"
        sx={{ textTransform: 'none' }}
      >
        Home
      </Button>
      <Button 
        color="inherit" 
        component={RouterLink} 
        to="/detect"
        sx={{ textTransform: 'none' }}
      >
        Detect
      </Button>
      <Button 
        color="inherit" 
        component={RouterLink} 
        to="/about"
        sx={{ textTransform: 'none' }}
      >
        About
      </Button>
    </Box>
  );
};

export default Navigation;