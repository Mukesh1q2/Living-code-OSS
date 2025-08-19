# Vidya Quantum Interface - Development Guide

## Overview

This is the React frontend for the Vidya quantum Sanskrit AI interface, built with Next.js, Three.js, and React Three Fiber. The interface provides a responsive, quantum-consciousness-inspired UI for interacting with Sanskrit AI.

## Architecture

### Core Technologies
- **Next.js 14** - React framework with App Router
- **React Three Fiber** - React renderer for Three.js
- **Three.js** - 3D graphics library
- **Framer Motion** - Animation library
- **Zustand** - State management
- **TypeScript** - Type safety

### Component Structure

```
components/
├── Core Components
│   ├── ResponsiveLayout.tsx     # Main responsive wrapper
│   ├── QuantumCanvas.tsx        # Three.js canvas with responsive settings
│   ├── Vidya.tsx               # Main consciousness display
│   └── DevTools.tsx            # Development tools integration
├── UI Components
│   ├── Header.tsx              # Responsive header
│   ├── HUD.tsx                 # Heads-up display
│   ├── Footer.tsx              # Footer component
│   ├── ChatInterface.tsx       # Chat interface
│   └── PlanPanel.tsx           # Planning panel
├── Quantum Components
│   ├── QuantumNetwork.tsx      # 3D neural network
│   ├── EntangledInstances.tsx  # Multiple Vidya instances
│   └── SuperpositionOverlay.tsx # Quantum effects
└── Development Tools
    ├── ThreeInspector.tsx      # Three.js scene inspector
    └── README.md               # Component documentation
```

## Responsive Design System

### Breakpoints
- **Mobile**: ≤ 480px
- **Tablet**: 481px - 768px  
- **Desktop**: 769px - 1024px
- **Wide**: ≥ 1440px

### Quantum Quality Levels
The interface adapts rendering quality based on device capabilities:

- **Minimal**: Basic wireframe rendering, reduced animations
- **Low**: Simple materials, limited particle effects
- **Medium**: Standard quality with most effects
- **High**: Full quality with advanced shaders and effects

### Usage Example
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

## Development Tools

### Built-in DevTools
- **Performance Monitor**: Real-time FPS and memory usage
- **Responsive Inspector**: Current breakpoint and device info
- **Quantum State Inspector**: Current quantum state values
- **Device Capabilities**: WebGL support, touch capabilities
- **Three.js Inspector**: Scene object browser with highlighting

### Accessing DevTools
1. **Toggle Panel**: Press `Ctrl + \`` or click the "DEV" button
2. **Stats Display**: Click "Toggle Stats" to show performance overlay
3. **Scene Inspector**: Click "Show Inspector" to browse Three.js objects
4. **React DevTools**: Install browser extension for React debugging

## Development Workflow

### Local Development
```bash
# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Lint code
npm run lint
```

### Testing Responsive Behavior
1. Use browser dev tools to test different screen sizes
2. Check the dev panel for current breakpoint information
3. Verify quantum quality adaptation on different devices
4. Test touch interactions on mobile devices

### Performance Optimization
1. Monitor FPS using the built-in stats display
2. Check Three.js object count in the scene inspector
3. Use quantum quality levels to optimize for different devices
4. Profile memory usage for complex scenes

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

## API Integration

### Chat Interface
The chat interface connects to backend APIs:
- `/api/analyze` - Sanskrit text analysis
- `/api/llm` - LLM completion requests

### State Management
Global state is managed with Zustand:
```tsx
import { useQuantumState } from '@/lib/state';

function MyComponent() {
  const quantumQuality = useQuantumState(s => s.quantumQuality);
  const setQuantumQuality = useQuantumState(s => s.setQuantumQuality);
  
  // Component logic
}
```

## Accessibility Features

### Touch-Friendly Design
- Minimum 44px touch targets
- Larger targets on mobile (48px)
- Touch gesture support for camera controls

### Reduced Motion Support
- Respects `prefers-reduced-motion` setting
- Minimal animations for accessibility

### High Contrast Support
- Adapts colors for `prefers-contrast: high`
- Maintains readability across themes

### Keyboard Navigation
- Full keyboard navigation support
- Focus indicators for interactive elements

## Troubleshooting

### Common Issues

1. **Three.js Performance**: 
   - Check quantum quality setting
   - Monitor object count in scene inspector
   - Reduce particle effects on mobile

2. **Responsive Layout Issues**:
   - Verify breakpoint detection in dev tools
   - Check CSS custom properties
   - Test on actual devices

3. **State Management**:
   - Use React DevTools to inspect state
   - Check Zustand store in dev panel
   - Verify state persistence

### Debug Tools
- Browser console for Three.js warnings
- React DevTools for component inspection
- Built-in dev panel for quantum state
- Network tab for API requests

## Future Enhancements

### Planned Features
- Voice interaction integration
- Advanced Three.js inspector with real-time editing
- Real-time Sanskrit morphological analysis display
- Enhanced quantum entanglement visualizations
- VR/AR support for immersive experiences

### Architecture Improvements
- Component lazy loading for better performance
- Advanced state management for complex quantum states
- WebWorker integration for heavy computations
- Service worker for offline capabilities

## Contributing

### Code Style
- Use TypeScript for all new components
- Follow responsive design patterns
- Include accessibility considerations
- Add performance monitoring where needed

### Testing
- Test all breakpoints using browser dev tools
- Verify quantum effects work across quality levels
- Check accessibility features
- Validate touch interactions on mobile devices

### Documentation
- Update component README when adding features
- Document responsive behavior patterns
- Include performance considerations
- Add accessibility notes