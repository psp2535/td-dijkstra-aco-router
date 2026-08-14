import React, { useState, useEffect, useRef, useMemo } from 'react';
import { MapContainer, TileLayer, Marker, Polyline, useMapEvents, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { MapPin, Navigation, Zap, Droplets, Wind, Clock, ChevronRight, Play, Pause, RotateCcw, FastForward } from 'lucide-react';
import axios from 'axios';
import L from 'leaflet';

// Fix generic leaflet icon
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const API_URL = import.meta.env.VITE_API_URL || '';

function MapClickHandler({ onMapClick }) {
  useMapEvents({
    click(e) {
      onMapClick(e.latlng);
    }
  });
  return null;
}

const FitBounds = ({ bounds }) => {
  const map = useMap();
  useEffect(() => {
    if (bounds) {
      map.fitBounds(bounds, { padding: [50, 50], maxZoom: 15, animate: true, duration: 1 });
    }
  }, [bounds, map]);
  return null;
};

const createNumberedIcon = (number, color, isDimmed) => {
  return L.divIcon({
    className: 'custom-number-icon',
    html: `<div class="marker-pin" style="background-color: ${color}; border-color: ${color}; opacity: ${isDimmed ? 0.4 : 1}; filter: ${isDimmed ? 'grayscale(100%)' : 'none'};">${number}</div>`,
    iconSize: [28, 28],
    iconAnchor: [14, 14]
  });
};

const createLabelIcon = (label, color, isDimmed) => {
  return L.divIcon({
    className: 'custom-label-icon',
    html: `<div class="marker-label" style="background-color: ${color}; opacity: ${isDimmed ? 0.4 : 1};">${label}</div>`,
    iconSize: [60, 28],
    iconAnchor: [30, 14]
  });
};

const createVehicleIcon = (color, bearing) => {
  return L.divIcon({
    className: 'vehicle-icon',
    html: `<div class="vehicle-shape" style="
      background-color: ${color}; 
      transform: rotate(${bearing}deg);
      width: 20px; 
      height: 20px; 
      border-radius: 50% 50% 0 0;
      border: 2px solid white; 
      box-shadow: 0 0 15px ${color};
      position: relative;
      transition: transform 0.1s linear;
    ">
      <div style="position: absolute; top: -5px; left: 6px; width: 4px; height: 4px; background: white; border-radius: 50%;"></div>
    </div>`,
    iconSize: [20, 20],
    iconAnchor: [10, 10]
  });
};

const MetricBlock = ({ icon: Icon, label, oldVal, newVal, unit, betterIsLower = true }) => {
  const isBetter = betterIsLower ? newVal < oldVal : newVal > oldVal;
  const pctDiff = oldVal === 0 ? 0 : Math.abs(((newVal - oldVal) / oldVal) * 100).toFixed(1);
  const color = isBetter ? '#10b981' : '#f43f5e';
  const displayVal = newVal % 1 === 0 ? newVal : newVal.toFixed(2);

  return (
    <div className="metric-block">
      <div className="metric-header">
        <Icon size={16} className="metric-icon" />
        <span className="metric-title">{label}</span>
      </div>
      <div className="metric-values">
        <div className="metric-newval">{displayVal} <span className="unit">{unit}</span></div>
        <div className="metric-diff" style={{ color }}>
          {isBetter ? '↓' : '↑'} {pctDiff}%
        </div>
      </div>
      <div className="metric-oldval">vs {oldVal.toFixed(2)} {unit}</div>
    </div>
  );
};

// --- SIMULATION LOGIC ---

function extractRouteState(segments, currentTime) {
  let timeAcc = 0;
  let completedCoords = [];
  let remainingCoords = [];
  let currentSegment = null;
  let currentCoord = null;
  let bearing = 0;
  
  if (!segments || segments.length === 0) return null;

  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    if (timeAcc + seg.duration_sec < currentTime) {
       // Completed segment
       timeAcc += seg.duration_sec;
       completedCoords.push(...seg.coords);
    } else if (timeAcc <= currentTime && currentSegment === null) {
       // Current segment
       currentSegment = seg;
       const timeInSeg = currentTime - timeAcc;
       const fraction = seg.duration_sec === 0 ? 0 : timeInSeg / seg.duration_sec;
       
       const totalPoints = seg.coords.length;
       if (totalPoints < 2) {
           currentCoord = seg.coords[0];
           completedCoords.push(currentCoord);
           remainingCoords.push(currentCoord);
       } else {
           // Distance-based interpolation for smooth movement across uneven geometry
           let totalDist = 0;
           const dists = [];
           for (let j = 0; j < totalPoints - 1; j++) {
               const p0 = seg.coords[j];
               const p1 = seg.coords[j+1];
               const d = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]);
               dists.push(d);
               totalDist += d;
           }
           
           const targetDist = fraction * totalDist;
           let accum = 0;
           let idx0 = 0;
           let segFraction = 0;
           
           for (let j = 0; j < dists.length; j++) {
               if (accum + dists[j] >= targetDist || j === dists.length - 1) {
                   idx0 = j;
                   segFraction = dists[j] === 0 ? 0 : (targetDist - accum) / dists[j];
                   // clamp segFraction just in case
                   segFraction = Math.max(0, Math.min(1, segFraction));
                   break;
               }
               accum += dists[j];
           }
           
           const idx1 = Math.min(idx0 + 1, totalPoints - 1);
           const p0 = seg.coords[idx0];
           const p1 = seg.coords[idx1];
           
           currentCoord = [
             p0[0] + (p1[0] - p0[0]) * segFraction,
             p0[1] + (p1[1] - p0[1]) * segFraction
           ];
           
           const dy = p1[0] - p0[0];
           const dx = p1[1] - p0[1];
           bearing = Math.atan2(dx, dy) * 180 / Math.PI;
           
           completedCoords.push(...seg.coords.slice(0, idx0 + 1), currentCoord);
           remainingCoords.push(currentCoord, ...seg.coords.slice(idx1));
       }
       timeAcc += seg.duration_sec;
    } else {
       // Remaining segment
       remainingCoords.push(...seg.coords);
    }
  }
  
  if (!currentCoord && completedCoords.length > 0) {
      currentCoord = completedCoords[completedCoords.length - 1];
      bearing = 0;
  }
  
  return { completedCoords, remainingCoords, currentCoord, currentSegment, bearing };
}

const AnimatedRoute = ({ segments, color, activeColor, dimColor, currentTime, totalDuration, isActive }) => {
  const state = useMemo(() => extractRouteState(segments, currentTime), [segments, currentTime]);
  
  if (!state || !state.currentCoord || !isActive) return null;
  
  const isFinished = currentTime >= totalDuration;
  
  return (
    <React.Fragment>
      {/* Completed Route */}
      {state.completedCoords.length > 1 && (
        <Polyline 
          positions={state.completedCoords} 
          color={dimColor} 
          weight={4} 
          opacity={0.8}
          lineCap="round"
          lineJoin="round"
        />
      )}
      
      {/* Current Segment (Glow) */}
      {!isFinished && state.currentSegment && (
        <Polyline 
          positions={state.currentSegment.coords} 
          color={color} 
          weight={8} 
          opacity={0.8}
          lineCap="round"
          lineJoin="round"
        />
      )}
      
      {/* Remaining Route */}
      {state.remainingCoords.length > 1 && (
        <Polyline 
          positions={state.remainingCoords} 
          color={color} 
          weight={3} 
          opacity={0.4}
          dashArray="5, 10"
          lineCap="round"
          lineJoin="round"
        />
      )}
      
      {/* Vehicle Marker */}
      <Marker 
        position={state.currentCoord} 
        icon={createVehicleIcon(activeColor, state.bearing)} 
        zIndexOffset={1000} 
      />
    </React.Fragment>
  );
};


function App() {
  const [points, setPoints] = useState([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [bounds, setBounds] = useState(null);
  
  // Simulation State
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [viewMode, setViewMode] = useState('both'); // 'both', 'old', 'new'
  
  const requestRef = useRef();
  const lastTimeRef = useRef();
  
  const totalDuration = useMemo(() => {
    if (!results) return 0;
    const oldDur = results.old_algo.metrics.duration_sec;
    const newDur = results.new_algo.metrics.duration_sec;
    return Math.max(oldDur, newDur);
  }, [results]);

  const animate = timeNow => {
    if (lastTimeRef.current !== undefined) {
      const delta = (timeNow - lastTimeRef.current) / 1000;
      // Standard 60x time lapse (1 minute real-time = 1 second simulation)
      // Cap it so it never goes blindingly fast.
      let speedMultiplier = 60.0; 
      
      // If the route is extremely long (>1 hour), bump it slightly so we don't wait forever, but cap at 120x
      if (totalDuration > 3600) speedMultiplier = 120.0;
      else if (totalDuration > 1800) speedMultiplier = 90.0;

      setCurrentTime(prev => {
         const next = prev + delta * speedMultiplier * playbackSpeed;
         if (next >= totalDuration) {
             setIsPlaying(false);
             return totalDuration;
         }
         return next;
      });
    }
    lastTimeRef.current = timeNow;
    if (isPlaying) {
      requestRef.current = requestAnimationFrame(animate);
    }
  };

  useEffect(() => {
    if (isPlaying) {
      requestRef.current = requestAnimationFrame(animate);
    } else {
      lastTimeRef.current = undefined;
    }
    return () => cancelAnimationFrame(requestRef.current);
  }, [isPlaying, playbackSpeed, totalDuration]);

  const handlePlayPause = () => {
      if (currentTime >= totalDuration) {
          setCurrentTime(0);
      }
      setIsPlaying(!isPlaying);
  }
  
  const handleReset = () => {
    setIsPlaying(false);
    setCurrentTime(0);
    lastTimeRef.current = undefined;
  };
  
  const handleMapClick = (latlng) => {
    if (points.length < 10) {
      setPoints([...points, { lat: latlng.lat, lon: latlng.lng }]);
    }
  };
  
  const handleClear = () => {
    setPoints([]);
    setResults(null);
    handleReset();
  };
  
  const handleCalculate = async () => {
    if (points.length < 2) return;
    
    setLoading(true);
    try {
      const depot = points[0];
      const deliveries = points.slice(1);
      
      const res = await axios.post(`${API_URL}/api/compare`, {
        depot,
        deliveries
      });
      
      setResults(res.data);
      
      // Calculate Bounds
      const allCoords = [];
      res.data.old_algo.segments.forEach(s => allCoords.push(...s.coords));
      res.data.new_algo.segments.forEach(s => allCoords.push(...s.coords));
      
      if (allCoords.length > 0) {
          const lats = allCoords.map(c => c[0]);
          const lons = allCoords.map(c => c[1]);
          setBounds([
              [Math.min(...lats), Math.min(...lons)],
              [Math.max(...lats), Math.max(...lons)]
          ]);
      }
      
      handleReset();
      setIsPlaying(true);
      
    } catch (err) {
      if (err.response && err.response.data && err.response.data.detail) {
          alert(err.response.data.detail);
      } else {
          alert("Failed to calculate routes. Ensure backend is running.");
      }
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const currentNewState = useMemo(() => results ? extractRouteState(results.new_algo.segments, currentTime) : null, [results, currentTime]);
  const currentOldState = useMemo(() => results ? extractRouteState(results.old_algo.segments, currentTime) : null, [results, currentTime]);

  return (
    <div className="app-container">
      {/* Sidebar Dashboard */}
      <div className="dashboard">
        <div className="dashboard-header">
          <div className="logo-area">
            <Zap size={24} color="#38bdf8" />
            <h1>FleetMaster <span>Pro</span></h1>
          </div>
          <p>Real-Time Telemetry & Route Operations</p>
        </div>
        
        <div className="controls-card glass-panel">
          <h3>Routing Input</h3>
          <div className="points-status">
            <div className="status-indicator">
              <span className="dot active"></span>
              {points.length} / 10 Depots Scheduled
            </div>
            {points.length === 0 && <span className="hint">Click map to add Start Depot</span>}
          </div>
          
          <div className="button-group">
            <button 
              className="btn btn-primary" 
              onClick={handleCalculate}
              disabled={points.length < 2 || loading}
            >
              {loading ? (
                <><span className="spinner"></span> Dispatching...</>
              ) : (
                <>Deploy Fleet <ChevronRight size={18} /></>
              )}
            </button>
            <button className="btn btn-clear" onClick={handleClear}>Reset</button>
          </div>
        </div>
        
        {results && !loading && (
          <div className="results-container">
            <div className="view-toggles" style={{display: 'flex', gap: '5px', marginBottom: '15px'}}>
                <button className={`btn ${viewMode === 'both' ? 'btn-primary' : 'btn-clear'}`} style={{flex: 1, padding: '8px'}} onClick={() => setViewMode('both')}>Both</button>
                <button className={`btn ${viewMode === 'new' ? 'btn-primary' : 'btn-clear'}`} style={{flex: 1, padding: '8px'}} onClick={() => setViewMode('new')}>TD+ACO</button>
                <button className={`btn ${viewMode === 'old' ? 'btn-primary' : 'btn-clear'}`} style={{flex: 1, padding: '8px'}} onClick={() => setViewMode('old')}>Static</button>
            </div>
          
            <div className="metrics-grid glass-panel active-panel">
              <MetricBlock 
                icon={Navigation} 
                label="Total Distance" 
                oldVal={results.old_algo.metrics.distance_km} 
                newVal={results.new_algo.metrics.distance_km} 
                unit="km" 
              />
              <MetricBlock 
                icon={Clock} 
                label="Est. Time" 
                oldVal={results.old_algo.metrics.duration_sec / 60} 
                newVal={results.new_algo.metrics.duration_sec / 60} 
                unit="min" 
              />
              <MetricBlock 
                icon={Droplets} 
                label="Fuel Cost" 
                oldVal={results.old_algo.metrics.fuel_l} 
                newVal={results.new_algo.metrics.fuel_l} 
                unit="L" 
              />
              <MetricBlock 
                icon={Wind} 
                label="Carbon Footprint" 
                oldVal={results.old_algo.metrics.co2_g / 1000} 
                newVal={results.new_algo.metrics.co2_g / 1000} 
                unit="kg" 
              />
            </div>
            
            <div className="playback-controls glass-panel" style={{marginTop: '15px', padding: '15px'}}>
               <h3 style={{marginTop: 0}}>Playback Center</h3>
               <div className="progress-bar" style={{height: '6px', background: '#334155', borderRadius: '3px', overflow: 'hidden', margin: '15px 0'}}>
                  <div className="progress-fill" style={{width: `${(currentTime / totalDuration) * 100}%`, height: '100%', background: '#38bdf8', transition: 'width 0.1s linear'}}></div>
               </div>
               <div className="control-buttons" style={{display: 'flex', justifyContent: 'center', gap: '15px'}}>
                  <button className="btn btn-clear" style={{padding: '8px'}} onClick={handleReset}><RotateCcw size={20}/></button>
                  <button className="btn btn-primary" style={{padding: '8px 16px', borderRadius: '20px'}} onClick={handlePlayPause}>
                     {isPlaying ? <Pause size={20}/> : <Play size={20}/>}
                  </button>
                  <button className="btn btn-clear" style={{padding: '8px'}} onClick={() => setPlaybackSpeed(s => s === 2 ? 0.5 : s === 0.5 ? 1 : 2)}>
                     <FastForward size={20}/> {playbackSpeed}x
                  </button>
               </div>
            </div>
          </div>
        )}
      </div>
      
      {/* Map */}
      <div className="map-container" style={{position: 'relative'}}>
        
        {/* Live Telemetry Overlay */}
        {results && (
           <div className="telemetry-overlay" style={{position: 'absolute', top: 20, right: 20, zIndex: 1000, display: 'flex', flexDirection: 'column', gap: '10px'}}>
              {(viewMode === 'both' || viewMode === 'new') && currentNewState && (
                  <div className="glass-panel" style={{borderLeft: '4px solid #10b981', padding: '10px 15px', minWidth: '220px', background: 'rgba(15, 23, 42, 0.85)', backdropFilter: 'blur(8px)'}}>
                      <h4 style={{margin: '0 0 10px 0', color: '#34d399'}}>New Algo (Green)</h4>
                      <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '5px'}}><span style={{color: '#94a3b8'}}>Road:</span> <strong style={{color: 'white', maxWidth: '120px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis'}}>{currentNewState.currentSegment?.name || 'Arrived'}</strong></div>
                      <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '5px'}}><span style={{color: '#94a3b8'}}>Speed:</span> <strong style={{color: 'white'}}>{currentNewState.currentSegment ? Math.round(currentNewState.currentSegment.speed_kmh) : 0} km/h</strong></div>
                      <div style={{display: 'flex', justifyContent: 'space-between'}}><span style={{color: '#94a3b8'}}>Progress:</span> <strong style={{color: 'white'}}>{Math.min(100, Math.round((currentTime / results.new_algo.metrics.duration_sec)*100))}%</strong></div>
                  </div>
              )}
              {(viewMode === 'both' || viewMode === 'old') && currentOldState && (
                  <div className="glass-panel" style={{borderLeft: '4px solid #e11d48', padding: '10px 15px', minWidth: '220px', background: 'rgba(15, 23, 42, 0.85)', backdropFilter: 'blur(8px)'}}>
                      <h4 style={{margin: '0 0 10px 0', color: '#fb7185'}}>Old Algo (Red)</h4>
                      <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '5px'}}><span style={{color: '#94a3b8'}}>Road:</span> <strong style={{color: 'white', maxWidth: '120px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis'}}>{currentOldState.currentSegment?.name || 'Arrived'}</strong></div>
                      <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '5px'}}><span style={{color: '#94a3b8'}}>Speed:</span> <strong style={{color: 'white'}}>{currentOldState.currentSegment ? Math.round(currentOldState.currentSegment.speed_kmh) : 0} km/h</strong></div>
                      <div style={{display: 'flex', justifyContent: 'space-between'}}><span style={{color: '#94a3b8'}}>Progress:</span> <strong style={{color: 'white'}}>{Math.min(100, Math.round((currentTime / results.old_algo.metrics.duration_sec)*100))}%</strong></div>
                  </div>
              )}
           </div>
        )}

        <MapContainer 
          center={[39.9, 116.4]} 
          zoom={11} 
          style={{ height: '100%', width: '100%', background: '#0b1120' }}
          wheelPxPerZoomLevel={120}
        >
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            attribution='&copy; CARTO'
          />
          <MapClickHandler onMapClick={handleMapClick} />
          {bounds && <FitBounds bounds={bounds} />}
          
          {/* User Input Points */}
          {!results && points.map((pt, i) => (
            <Marker 
              key={`pt-${i}`} 
              position={[pt.lat, pt.lon]} 
              icon={i === 0 ? createLabelIcon("START", "#3b82f6", false) : createNumberedIcon(i, "#64748b", false)} 
            />
          ))}

          {/* AI Result Routes */}
          {results && (
            <React.Fragment>
              
              {/* Old Route */}
              <AnimatedRoute 
                 segments={results.old_algo.segments}
                 color="#e11d48"
                 activeColor="#f43f5e"
                 dimColor="#881337"
                 currentTime={currentTime}
                 totalDuration={results.old_algo.metrics.duration_sec}
                 isActive={viewMode === 'both' || viewMode === 'old'}
              />
              
              {/* New Route */}
              <AnimatedRoute 
                 segments={results.new_algo.segments}
                 color="#10b981"
                 activeColor="#34d399"
                 dimColor="#064e3b"
                 currentTime={currentTime}
                 totalDuration={results.new_algo.metrics.duration_sec}
                 isActive={viewMode === 'both' || viewMode === 'new'}
              />
              
              {points.map((pt, i) => {
                 if (i === 0) {
                     return <Marker key={`res-pt-${i}`} position={[pt.lat, pt.lon]} icon={createLabelIcon("START / END", "#3b82f6", false)} />
                 }
                 
                 // Determine visit order for this point based on active view mode
                 let activeAlgo = viewMode === 'old' ? results.old_algo : results.new_algo;
                 // ordered_indices maps: visit step -> delivery index
                 // Example: [0, 2, 1, 0] means Visit 1 is delivery 2 (points[2])
                 let visitOrder = activeAlgo.ordered_indices.indexOf(i);
                 if (visitOrder === -1) visitOrder = i; // Fallback
                 
                 return (
                 <Marker 
                   key={`res-pt-${i}`} 
                   position={[pt.lat, pt.lon]} 
                   icon={createNumberedIcon(visitOrder, viewMode === 'old' ? "#e11d48" : "#10b981", false)} 
                 />
                 )
              })}
            </React.Fragment>
          )}
        </MapContainer>
      </div>
    </div>
  );
}

export default App;
