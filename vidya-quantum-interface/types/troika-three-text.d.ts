declare module 'troika-three-text' {
  import * as THREE from 'three';

  export class Text extends THREE.Mesh {
    text: string;
    fontSize: number;
    color: THREE.Color | string | number;
    anchorX: 'left' | 'center' | 'right' | number;
    anchorY: 'top' | 'top-baseline' | 'middle' | 'bottom-baseline' | 'bottom' | number;
    font: string;
    fontWeight: string | number;
    fontStyle: string;
    letterSpacing: number;
    lineHeight: number | string;
    maxWidth: number;
    overflowWrap: 'normal' | 'break-word';
    textAlign: 'left' | 'right' | 'center' | 'justify';
    textIndent: number;
    whiteSpace: 'normal' | 'nowrap';
    direction: 'auto' | 'ltr' | 'rtl';
    clipRect: [number, number, number, number] | null;
    depthOffset: number;
    curveRadius: number;
    debugSDF: boolean;
    sdfGlyphSize: number;
    gpuAccelerateSDF: boolean;
    
    constructor();
    
    sync(callback?: () => void): void;
    dispose(): void;
    
    // Getters for computed properties
    readonly textRenderInfo: {
      parameters: any;
      sdfTexture: THREE.Texture;
      blockBounds: [number, number, number, number];
      visibleBounds: [number, number, number, number];
      chunkedBounds: [number, number, number, number][];
      timings: { [key: string]: number };
    } | null;
  }
}