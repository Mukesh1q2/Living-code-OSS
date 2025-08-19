"use client";

import React, { useState, useMemo } from 'react';
import { useLearningAnalytics, usePersonalityEvolution, useComplexityScaling } from '@/lib/useAdvancedLearning';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';

interface LearningAnalyticsProps {
  className?: string;
  showDetailed?: boolean;
}

export default function LearningAnalytics({ className = "", showDetailed = false }: LearningAnalyticsProps) {
  const analytics = useLearningAnalytics();
  const { evolutionData, totalEvolutions, mostActiveTraits } = usePersonalityEvolution();
  const { complexityData, isGrowthNeeded } = useComplexityScaling();
  const [selectedTab, setSelectedTab] = useState('overview');

  // Prepare chart data
  const personalityChartData = useMemo(() => {
    return evolutionData.map(evolution => ({
      name: evolution.traitName,
      current: evolution.currentValue,
      original: evolution.originalValue,
      trend: evolution.trend,
      activity: evolution.recentActivity
    }));
  }, [evolutionData]);

  const complexityChartData = useMemo(() => {
    return [
      { name: 'Sanskrit', value: complexityData.sanskritSophistication * 100 },
      { name: 'Neural Network', value: complexityData.neuralNetworkDensity * 100 },
      { name: 'Response', value: complexityData.responseComplexity * 100 },
      { name: 'Quantum', value: complexityData.quantumCoherenceLevel * 100 }
    ];
  }, [complexityData]);

  const preferencesChartData = useMemo(() => {
    return analytics.mostActivePreferences.map(pref => ({
      name: pref.preference,
      strength: pref.strength * 100,
      confidence: pref.confidence * 100,
      sessions: pref.sessionCount
    }));
  }, [analytics.mostActivePreferences]);

  const milestonesData = useMemo(() => {
    return analytics.recentMilestones.map((milestone, index) => ({
      id: index,
      type: milestone.type,
      description: milestone.description,
      significance: milestone.significance * 100,
      date: milestone.timestamp.toLocaleDateString(),
      impact: milestone.impact.join(', ')
    }));
  }, [analytics.recentMilestones]);

  const COLORS = ['#7BE1FF', '#B383FF', '#63FFC9', '#FFB366', '#FF8A80'];

  return (
    <div className={`learning-analytics ${className}`}>
      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="personality">Personality</TabsTrigger>
          <TabsTrigger value="complexity">Complexity</TabsTrigger>
          <TabsTrigger value="preferences">Preferences</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Interactions</CardTitle>
                <span className="text-2xl">🧠</span>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{analytics.interactionPatternCount}</div>
                <p className="text-xs text-muted-foreground">
                  Learning velocity: {(analytics.learningVelocity * 100).toFixed(1)}%
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Personality Evolutions</CardTitle>
                <span className="text-2xl">🌱</span>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{totalEvolutions}</div>
                <p className="text-xs text-muted-foreground">
                  Stability: {(analytics.personalityStability * 100).toFixed(0)}%
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Complexity Level</CardTitle>
                <span className="text-2xl">⚡</span>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{complexityData.currentLevel.toFixed(1)}</div>
                <p className="text-xs text-muted-foreground">
                  Growth rate: {(complexityData.evolutionRate * 100).toFixed(1)}%
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Adaptation Success</CardTitle>
                <span className="text-2xl">🎯</span>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{(analytics.adaptationSuccessRate * 100).toFixed(0)}%</div>
                <p className="text-xs text-muted-foreground">
                  Efficiency: {analytics.learningEfficiency.toFixed(2)}
                </p>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Recent Learning Milestones</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {milestonesData.slice(0, 5).map((milestone) => (
                  <div key={milestone.id} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <Badge className="text-xs border border-gray-300">
                          {milestone.type.replace('_', ' ')}
                        </Badge>
                        <span className="text-sm font-medium">{milestone.description}</span>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        Impact: {milestone.impact} • {milestone.date}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-bold">{milestone.significance.toFixed(0)}%</div>
                      <div className="text-xs text-muted-foreground">significance</div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Most Active Traits</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {mostActiveTraits.map((trait, index) => (
                    <div key={trait} className="flex items-center justify-between">
                      <span className="text-sm capitalize">{trait}</span>
                      <Badge className="bg-gray-200 text-gray-800">#{index + 1}</Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Learning Status</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Learning Efficiency</span>
                    <span>{(analytics.learningEfficiency * 100).toFixed(0)}%</span>
                  </div>
                  <Progress value={analytics.learningEfficiency * 100} className="h-2" />
                </div>
                
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Personality Stability</span>
                    <span>{(analytics.personalityStability * 100).toFixed(0)}%</span>
                  </div>
                  <Progress value={analytics.personalityStability * 100} className="h-2" />
                </div>
                
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Preference Strength</span>
                    <span>{(analytics.preferenceStrength * 100).toFixed(0)}%</span>
                  </div>
                  <Progress value={analytics.preferenceStrength * 100} className="h-2" />
                </div>
                
                {isGrowthNeeded && (
                  <Badge className="w-full justify-center bg-red-500 text-white">
                    Complexity Growth Needed
                  </Badge>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="personality" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Personality Evolution Over Time</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={personalityChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis domain={[0, 1]} />
                  <Tooltip 
                    formatter={(value: number, name: string) => [
                      `${(value * 100).toFixed(1)}%`, 
                      name === 'current' ? 'Current Value' : 'Original Value'
                    ]}
                  />
                  <Bar dataKey="original" fill="#8884d8" name="original" />
                  <Bar dataKey="current" fill="#82ca9d" name="current" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Personality Trends</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {personalityChartData.map((trait) => (
                    <div key={trait.name} className="flex items-center justify-between p-2 border rounded">
                      <div>
                        <div className="font-medium capitalize">{trait.name}</div>
                        <div className="text-sm text-muted-foreground">
                          Current: {(trait.current * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`text-sm font-bold ${trait.trend > 0 ? 'text-green-600' : trait.trend < 0 ? 'text-red-600' : 'text-gray-600'}`}>
                          {trait.trend > 0 ? '↗' : trait.trend < 0 ? '↘' : '→'} {Math.abs(trait.trend * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-muted-foreground">
                          Activity: {trait.activity}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Evolution Activity</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie
                      data={personalityChartData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, activity }) => `${name}: ${activity}`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="activity"
                    >
                      {personalityChartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="complexity" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Complexity Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={complexityChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip formatter={(value: number) => [`${value.toFixed(1)}%`, 'Level']} />
                    <Bar dataKey="value" fill="#7BE1FF" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Complexity Progress</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span>Overall Level</span>
                    <span>{complexityData.currentLevel.toFixed(1)} / 10</span>
                  </div>
                  <Progress value={(complexityData.currentLevel / 10) * 100} className="h-3" />
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span>Sophistication Index</span>
                    <span>{(complexityData.sophisticationIndex * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={complexityData.sophisticationIndex * 100} className="h-3" />
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span>Next Milestone Progress</span>
                    <span>{complexityData.progressToNextMilestone.toFixed(0)}%</span>
                  </div>
                  <Progress value={complexityData.progressToNextMilestone} className="h-3" />
                </div>

                <div className="pt-2 border-t">
                  <div className="text-sm text-muted-foreground">
                    Growth Potential: {complexityData.growthPotential.toFixed(1)} levels
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Evolution Rate: {(complexityData.evolutionRate * 100).toFixed(1)}%
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Complexity Breakdown</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                {complexityChartData.map((metric, index) => (
                  <div key={metric.name} className="text-center p-4 border rounded-lg">
                    <div className="text-2xl font-bold" style={{ color: COLORS[index] }}>
                      {metric.value.toFixed(0)}%
                    </div>
                    <div className="text-sm text-muted-foreground">{metric.name}</div>
                    <Progress 
                      value={metric.value} 
                      className="h-2 mt-2" 
                      style={{ '--progress-background': COLORS[index] } as any}
                    />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="preferences" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>User Preferences</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={preferencesChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip 
                    formatter={(value: number, name: string) => [
                      `${value.toFixed(1)}%`, 
                      name === 'strength' ? 'Strength' : 'Confidence'
                    ]}
                  />
                  <Bar dataKey="strength" fill="#63FFC9" name="strength" />
                  <Bar dataKey="confidence" fill="#B383FF" name="confidence" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Preference Details</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {analytics.mostActivePreferences.map((pref, index) => (
                    <div key={`${pref.category}-${pref.preference}`} className="p-3 border rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <Badge className="text-xs border border-gray-300">
                          {pref.category.replace('_', ' ')}
                        </Badge>
                        <span className="text-sm font-medium">{(pref.strength * 100).toFixed(0)}%</span>
                      </div>
                      <div className="text-sm font-medium capitalize">{pref.preference}</div>
                      <div className="text-xs text-muted-foreground mt-1">
                        Confidence: {(pref.confidence * 100).toFixed(0)}% • 
                        Sessions: {pref.sessionCount} • 
                        Updated: {pref.lastUpdated.toLocaleDateString()}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Preference Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie
                      data={preferencesChartData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, strength }) => `${name}: ${strength.toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="strength"
                    >
                      {preferencesChartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}