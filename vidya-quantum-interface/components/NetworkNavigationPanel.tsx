"use client";

import { useState, useEffect } from "react";
import { NetworkSection } from "@/lib/network-interactions";
import { useResponsive } from "@/lib/responsive";
import { useQuantumState } from "@/lib/state";

interface NetworkNavigationPanelProps {
  sections: NetworkSection[];
  currentSection: NetworkSection | null;
  onSectionSelect: (sectionId: string) => void;
  onClose?: () => void;
  visible?: boolean;
}

export default function NetworkNavigationPanel({
  sections,
  currentSection,
  onSectionSelect,
  onClose,
  visible = true
}: NetworkNavigationPanelProps) {
  const { isMobile, isTablet } = useResponsive();
  const [isExpanded, setIsExpanded] = useState(!isMobile);
  const selectedNode = useQuantumState((s) => s.selectedNode);

  useEffect(() => {
    // Auto-collapse on mobile when a node is selected
    if (isMobile && selectedNode) {
      setIsExpanded(false);
    }
  }, [isMobile, selectedNode]);

  if (!visible) return null;

  const panelStyles: React.CSSProperties = {
    position: 'fixed',
    top: isMobile ? '60px' : '80px',
    left: isMobile ? '10px' : '20px',
    width: isMobile ? (isExpanded ? '280px' : '50px') : '320px',
    maxHeight: isMobile ? '60vh' : '70vh',
    background: 'rgba(0, 0, 0, 0.9)',
    border: '1px solid rgba(123, 225, 255, 0.3)',
    borderRadius: '12px',
    backdropFilter: 'blur(8px)',
    zIndex: 1000,
    transition: 'all 0.3s ease',
    overflow: 'hidden',
    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.6)',
  };

  const headerStyles: React.CSSProperties = {
    padding: isMobile ? '12px' : '16px',
    borderBottom: isExpanded ? '1px solid rgba(123, 225, 255, 0.2)' : 'none',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    cursor: isMobile ? 'pointer' : 'default',
  };

  const contentStyles: React.CSSProperties = {
    maxHeight: isExpanded ? '400px' : '0',
    overflow: 'auto',
    transition: 'max-height 0.3s ease',
    padding: isExpanded ? (isMobile ? '8px' : '12px') : '0',
  };

  return (
    <div style={panelStyles}>
      {/* Header */}
      <div 
        style={headerStyles}
        onClick={isMobile ? () => setIsExpanded(!isExpanded) : undefined}
      >
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          <div style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: currentSection ? '#4AFF4A' : '#7BE1FF',
            boxShadow: `0 0 8px ${currentSection ? '#4AFF4A' : '#7BE1FF'}`,
          }} />
          {(!isMobile || isExpanded) && (
            <div style={{
              color: '#7BE1FF',
              fontSize: isMobile ? '14px' : '16px',
              fontWeight: 'bold',
              fontFamily: 'monospace'
            }}>
              Network Sections
            </div>
          )}
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {(!isMobile || isExpanded) && currentSection && (
            <div style={{
              background: 'rgba(74, 255, 74, 0.2)',
              padding: '2px 8px',
              borderRadius: '12px',
              fontSize: '10px',
              color: '#4AFF4A',
              textTransform: 'uppercase',
              fontWeight: 'bold'
            }}>
              Active
            </div>
          )}
          
          {isMobile && (
            <div style={{
              color: '#7BE1FF',
              fontSize: '12px',
              transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.3s ease'
            }}>
              ▼
            </div>
          )}
          
          {onClose && (!isMobile || isExpanded) && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onClose();
              }}
              style={{
                background: 'rgba(255, 100, 100, 0.2)',
                border: '1px solid rgba(255, 100, 100, 0.3)',
                borderRadius: '4px',
                color: '#ff6464',
                padding: '4px 8px',
                fontSize: '10px',
                cursor: 'pointer'
              }}
            >
              ×
            </button>
          )}
        </div>
      </div>

      {/* Content */}
      <div style={contentStyles}>
        {sections.map((section) => (
          <SectionItem
            key={section.id}
            section={section}
            isActive={currentSection?.id === section.id}
            onSelect={() => onSectionSelect(section.id)}
            isMobile={isMobile}
          />
        ))}
        
        {isExpanded && (
          <div style={{
            marginTop: '16px',
            padding: '12px',
            background: 'rgba(123, 225, 255, 0.05)',
            borderRadius: '8px',
            border: '1px solid rgba(123, 225, 255, 0.1)'
          }}>
            <div style={{
              color: '#7BE1FF',
              fontSize: '12px',
              fontWeight: 'bold',
              marginBottom: '8px'
            }}>
              Navigation Help
            </div>
            <div style={{
              color: '#7BE1FF',
              fontSize: '10px',
              lineHeight: '1.4',
              opacity: 0.8
            }}>
              <div>• Click sections to navigate</div>
              <div>• Use Tab/Shift+Tab to cycle nodes</div>
              <div>• Arrow keys for connected navigation</div>
              <div>• Double-click nodes to focus</div>
              {isMobile && <div>• Touch and hold for tooltips</div>}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function SectionItem({
  section,
  isActive,
  onSelect,
  isMobile
}: {
  section: NetworkSection;
  isActive: boolean;
  onSelect: () => void;
  isMobile: boolean;
}) {
  const [isHovered, setIsHovered] = useState(false);

  const itemStyles: React.CSSProperties = {
    padding: isMobile ? '10px' : '12px',
    marginBottom: '8px',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    border: isActive ? '1px solid rgba(123, 225, 255, 0.5)' : '1px solid transparent',
    background: isActive 
      ? 'rgba(123, 225, 255, 0.1)' 
      : isHovered 
        ? 'rgba(123, 225, 255, 0.05)' 
        : 'transparent',
  };

  const getSectionColor = (sectionId: string): string => {
    switch (sectionId) {
      case 'sanskrit-rules':
        return '#FFD700';
      case 'neural-units':
        return '#7BE1FF';
      case 'quantum-gates':
        return '#B383FF';
      default:
        return '#7BE1FF';
    }
  };

  const getSectionIcon = (sectionId: string): string => {
    switch (sectionId) {
      case 'sanskrit-rules':
        return '🕉️';
      case 'neural-units':
        return '🧠';
      case 'quantum-gates':
        return '⚛️';
      default:
        return '🔗';
    }
  };

  const sectionColor = getSectionColor(section.id);

  return (
    <div
      style={itemStyles}
      onClick={onSelect}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '6px'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          <span style={{ fontSize: isMobile ? '14px' : '16px' }}>
            {getSectionIcon(section.id)}
          </span>
          <div style={{
            color: sectionColor,
            fontSize: isMobile ? '12px' : '14px',
            fontWeight: 'bold'
          }}>
            {section.name}
          </div>
        </div>
        
        <div style={{
          background: `rgba(${sectionColor === '#FFD700' ? '255, 215, 0' : sectionColor === '#B383FF' ? '179, 131, 255' : '123, 225, 255'}, 0.2)`,
          padding: '2px 6px',
          borderRadius: '12px',
          fontSize: '9px',
          color: sectionColor,
          fontWeight: 'bold'
        }}>
          {section.nodes.length}
        </div>
      </div>
      
      <div style={{
        color: '#7BE1FF',
        fontSize: isMobile ? '10px' : '11px',
        opacity: 0.8,
        lineHeight: '1.3'
      }}>
        {section.description}
      </div>
      
      {isActive && (
        <div style={{
          marginTop: '8px',
          padding: '6px',
          background: 'rgba(123, 225, 255, 0.1)',
          borderRadius: '4px',
          fontSize: '9px',
          color: '#7BE1FF',
          opacity: 0.7
        }}>
          Position: ({section.centerPosition.x.toFixed(1)}, {section.centerPosition.y.toFixed(1)}, {section.centerPosition.z.toFixed(1)})
        </div>
      )}
    </div>
  );
}