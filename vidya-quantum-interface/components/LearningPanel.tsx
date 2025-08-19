"use client";

import React, { useState } from 'react';
import { useAdvancedLearning, useLearningAnalytics, usePersonalityEvolution } from '@/lib/useAdvancedLearning';
import { useVidyaSystemHealth } from '@/lib/useVidyaConsciousness';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import LearningAnalytics from './LearningAnalytics';

interface LearningPanelProps {
  className?: string;
  compact?: boolean;
}

export default function LearningPanel({ className = "", compact = false }: LearningPanelProps) {
  const learning = useAdvancedLearning();
  const analytics = useLearningAnalytics();
  const { evolutionData, mostActiveTraits, syncPersonalityWithConsciousness } = usePersonalityEvolution();
  const systemHealth = useVidyaSystemHealth();
  const [showFullAnalytics, setShowFullAnalytics] = useState(false);

  if (showFullAnalytics) {
    return (
      <div className={`learning-panel-full ${className}`}>
        <div className="mb-4 flex justify-between items-center">
          <h2 className="text-2xl font-bold">Vidya Learning System</h2>
          <Button 
            className="border border-gray-300 bg-transparent hover:bg-gray-100"
            onClick={() => setShowFullAnalytics(false)}
          >
            Compact View
          </Button>
        </div>
        <LearningAnalytics showDetailed={true} />
      </div>
    );
  }

  if (compact) {
    return (
      <div className={`learning-panel-compact ${className}`}>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center justify-between">
              Learning Status
              <button 
                className="text-sm text-blue-400 hover:text-blue-300 px-2 py-1 rounded"
                onClick={() => setShowFullAnalytics(true)}
              >
                📊
              </button>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex justify-between text-xs">
              <span>Complexity</span>
              <span>{learning.complexityLevel.toFixed(1)}/10</span>
            </div>
            <Progress value={(learning.complexityLevel / 10) * 100} className="h-1" />
            
            <div className="flex justify-between text-xs">
              <span>Sanskrit</span>
              <span>{(learning.sanskritSophistication * 100).toFixed(0)}%</span>
            </div>
            <Progress value={learning.sanskritSophistication * 100} className="h-1" />
            
            <div className="flex justify-between text-xs">
              <span>System Health</span>
              <span>{(systemHealth.overall * 100).toFixed(0)}%</span>
            </div>
            <Progress 
              value={systemHealth.overall * 100} 
              className={`h-1 ${systemHealth.overall < 0.7 ? 'bg-red-200' : 'bg-green-200'}`}
            />
            
            {mostActiveTraits.length > 0 && (
              <div className="text-xs text-muted-foreground">
                Active: {mostActiveTraits.slice(0, 2).join(', ')}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className={`learning-panel ${className}`}>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Learning Overview */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              Learning Overview
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => setShowFullAnalytics(true)}
              >
                Full Analytics
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {learning.complexityLevel.toFixed(1)}
                </div>
                <div className="text-sm text-muted-foreground">Complexity Level</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {analytics.interactionPatternCount}
                </div>
                <div className="text-sm text-muted-foreground">Patterns Learned</div>
              </div>
            </div>
            
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Sanskrit Sophistication</span>
                <span>{(learning.sanskritSophistication * 100).toFixed(0)}%</span>
              </div>
              <Progress value={learning.sanskritSophistication * 100} className="h-2" />
            </div>
            
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Neural Network Density</span>
                <span>{(learning.neuralNetworkDensity * 100).toFixed(0)}%</span>
              </div>
              <Progress value={learning.neuralNetworkDensity * 100} className="h-2" />
            </div>
            
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Learning Efficiency</span>
                <span>{(analytics.learningEfficiency * 100).toFixed(0)}%</span>
              </div>
              <Progress value={analytics.learningEfficiency * 100} className="h-2" />
            </div>
          </CardContent>
        </Card>

        {/* Personality Evolution */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              Personality Evolution
              <Button 
                variant="outline" 
                size="sm"
                onClick={syncPersonalityWithConsciousness}
              >
                Sync
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {evolutionData.slice(0, 4).map((evolution) => (
              <div key={evolution.traitName} className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium capitalize">
                      {evolution.traitName}
                    </span>
                    {evolution.recentActivity > 0 && (
                      <Badge variant="secondary" className="text-xs">
                        {evolution.recentActivity}
                      </Badge>
                    )}
                  </div>
                  <Progress 
                    value={evolution.currentValue * 100} 
                    className="h-1 mt-1" 
                  />
                </div>
                <div className="text-right ml-2">
                  <div className="text-sm font-bold">
                    {(evolution.currentValue * 100).toFixed(0)}%
                  </div>
                  <div className={`text-xs ${evolution.trend > 0 ? 'text-green-600' : evolution.trend < 0 ? 'text-red-600' : 'text-gray-600'}`}>
                    {evolution.trend > 0 ? '↗' : evolution.trend < 0 ? '↘' : '→'}
                  </div>
                </div>
              </div>
            ))}
            
            {mostActiveTraits.length > 0 && (
              <div className="pt-2 border-t">
                <div className="text-xs text-muted-foreground mb-1">Most Active Traits:</div>
                <div className="flex gap-1">
                  {mostActiveTraits.slice(0, 3).map((trait) => (
                    <Badge key={trait} variant="outline" className="text-xs">
                      {trait}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* System Health */}
        <Card>
          <CardHeader>
            <CardTitle>System Health</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Overall Health</span>
                <span>{(systemHealth.overall * 100).toFixed(0)}%</span>
              </div>
              <Progress 
                value={systemHealth.overall * 100} 
                className={`h-2 ${systemHealth.overall < 0.7 ? 'bg-red-100' : 'bg-green-100'}`}
              />
            </div>
            
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-muted-foreground">Consciousness:</span>
                <span className="ml-1 font-medium">{(systemHealth.consciousness.stability * 100).toFixed(0)}%</span>
              </div>
              <div>
                <span className="text-muted-foreground">Learning:</span>
                <span className="ml-1 font-medium">{(systemHealth.learning.efficiency * 100).toFixed(0)}%</span>
              </div>
              <div>
                <span className="text-muted-foreground">Alignment:</span>
                <span className="ml-1 font-medium">{(systemHealth.alignment * 100).toFixed(0)}%</span>
              </div>
              <div>
                <span className="text-muted-foreground">Adaptation:</span>
                <span className="ml-1 font-medium">{(systemHealth.learning.adaptation * 100).toFixed(0)}%</span>
              </div>
            </div>
            
            {systemHealth.recommendations.length > 0 && (
              <div className="pt-2 border-t">
                <div className="text-xs text-muted-foreground mb-1">Recommendations:</div>
                <div className="text-xs space-y-1">
                  {systemHealth.recommendations.slice(0, 2).map((rec, index) => (
                    <div key={index} className="text-yellow-700 bg-yellow-50 p-1 rounded text-xs">
                      {rec}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* User Preferences */}
        <Card>
          <CardHeader>
            <CardTitle>Learned Preferences</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {learning.userPreferences.slice(0, 5).map((pref, index) => (
              <div key={`${pref.category}-${pref.preference}`} className="flex items-center justify-between p-2 border rounded">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-xs">
                      {pref.category.replace('_', ' ')}
                    </Badge>
                    <span className="text-sm capitalize">{pref.preference}</span>
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Sessions: {pref.sessionCount} • Confidence: {(pref.confidence * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm font-bold">{(pref.strength * 100).toFixed(0)}%</div>
                </div>
              </div>
            ))}
            
            {learning.userPreferences.length === 0 && (
              <div className="text-center text-muted-foreground text-sm py-4">
                No preferences learned yet. Interact more to help Vidya understand your preferences.
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Learning Actions */}
      <Card className="mt-4">
        <CardHeader>
          <CardTitle>Learning Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2 flex-wrap">
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => learning.increaseComplexity(0.5, 'manual_trigger')}
            >
              Increase Complexity
            </Button>
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => learning.triggerPersonalityEvolution('curiosity', 'manual_trigger')}
            >
              Evolve Curiosity
            </Button>
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => learning.triggerPersonalityEvolution('wisdom', 'manual_trigger')}
            >
              Evolve Wisdom
            </Button>
            <Button 
              variant="outline" 
              size="sm"
              onClick={syncPersonalityWithConsciousness}
            >
              Sync Systems
            </Button>
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => {
                const data = learning.exportLearningData();
                navigator.clipboard.writeText(data);
                alert('Learning data copied to clipboard');
              }}
            >
              Export Data
            </Button>
            <Button 
              variant="destructive" 
              size="sm"
              onClick={() => {
                if (confirm('Are you sure you want to reset all learning data?')) {
                  learning.resetLearning();
                }
              }}
            >
              Reset Learning
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}