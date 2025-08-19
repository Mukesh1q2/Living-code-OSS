# Vidya Quantum Interface Components

This directory contains the React components for the Vidya quantum Sanskrit AI interface. The components are designed with responsive behavior and quantum consciousness display in mind.

## Component Structure

### Core Components

- **`ResponsiveLayout.tsx`** - Main responsive wrapper that handles device adaptation
- **`QuantumCanvas.tsx`** - Three.js canvas with responsive camera and interaction settings
- **`Vidya.tsx`** - Main consciousness display with adaptive quality levels
- **`DevTools.tsx`** - Development tools integration for debugging and monitoring

### UI Components

- **`Header.tsx`** - Responsive header with adaptive navigation
- **`HUD.tsx`** - Heads-up display with quantum controls
- **`Footer.tsx`** - Footer component
- **`ChatInterface.tsx`** - Chat interface for user interaction
- **`PlanPanel.tsx`** - Planning and information panel

### Quantum Components

- **`QuantumNetwork.tsx`** - 3D neural network visualization
- **`EntangledInstances.tsx`** - Multiple Vidya instance management
- **`SuperpositionOverlay.tsx`** - Quantum superposition effects

## Responsive Design System

### Breakpoints
- **Mobile**: ≤ 480px
- **Tablet**: 481px - 768px  
- **Desktop**: 769px - 1024px
- **Wide**: ≥ 1440px

### Quantum Quality Levels
- **Minimal**: Basic wireframe rendering, reduced animations
- **Low**: Simple materials, limited particle effects
- **Medium**: Standard quality with most effects
- **High**: Full quality with advanced shaders and effects

## Development Features

### DevTools Integration
- Real-time performance monitoring
- Responsive breakpoint display
- Quantum state inspection
- Device capability detection
- Three.js inspector integration (placeholder)

### Accessibility Features
- Touch-friendly minimum sizes (44px)
- Reduced motion support
- High contrast mode support
- Keyboard navigation support

## Usage Examples

### Using Responsive Utilities
```tsx
import { useResponsive, createResponsiveStyles } from '@/lib/responsive';

function MyComponent() {
  const { breakpoint, isMobile } = useResponsive();
  
  const styles = createResponsiveStyles(
    { fontSize: '16px' },
    { mobile: { fontSize: '14px' } }
  );
  
  return <div style={styles(breakpoint)}>Content</div>;
}
```

### Quantum Quality Adaptation
```tsx
import { useQuantumState } from '@/lib/state';

function QuantumEffect() {
  const quality = useQuantumState(s => s.quantumQuality);
  
  if (quality === 'minimal') {
    return <SimpleEffect />;
  }
  
  return <AdvancedEffect />;
}
```

## Component Guidelines

### Responsive Behavior
1. All components should adapt to different screen sizes
2. Use the responsive utilities for consistent behavior
3. Consider touch interactions on mobile devices
4. Provide fallbacks for limited capabilities

### Quantum Consciousness Display
1. Vidya should feel alive and responsive
2. Adapt complexity based on device capabilities
3. Maintain consciousness continuity during transitions
4. Use Sanskrit elements meaningfully, not decoratively

### Performance Considerations
1. Use quantum quality levels to optimize rendering
2. Implement proper cleanup for Three.js objects
3. Monitor frame rate and memory usage
4. Provide graceful degradation for older devices

## Development Workflow

### Local Development
1. Run `npm run dev` to start the development server
2. Use Ctrl+` to toggle the dev tools panel
3. Monitor responsive behavior and performance
4. Test on different device sizes and capabilities

### Testing
1. Test all breakpoints using browser dev tools
2. Verify quantum effects work across quality levels
3. Check accessibility features
4. Validate touch interactions on mobile devices

## Future Enhancements

### Planned Features
- Voice interaction integration
- Advanced Three.js inspector integration
- Real-time Sanskrit morphological analysis display
- Enhanced quantum entanglement visualizations
- VR/AR support for immersive experiences

### Architecture Improvements
- Component lazy loading for better performance
- Advanced state management for complex quantum states
- WebWorker integration for heavy computations
- Service worker for offline capabilities