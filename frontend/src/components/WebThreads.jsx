import { useEffect, useRef } from 'react';
import { Renderer, Program, Mesh, Triangle } from 'ogl';
import './WebThreads.css';

const hexToRgb = hex => {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? [1, 2, 3].map(i => parseInt(result[i], 16) / 255) : [1, 1, 1];
};
const FAN_MODE = { center: 0, left: 1, right: 2 };
const vertex = `#version 300 es
in vec2 position;
void main(){gl_Position=vec4(position,0.,1.);}`;
const fragment = `#version 300 es
precision highp float;
uniform vec2 iResolution,uMouse;
uniform float iTime,uSpeed,uThreadCount,uFrequency,uSpread,uTaper,uPosition,uFanMode,uGlow,uFalloff,uThickness,uBrightness,uOpacity,uMirror,uShimmer,uGrain,uGrainIntensity,uMouseStrength,uEnableMouse,uMouseActive;
uniform vec3 uColor1,uColor2,uColor3;
out vec4 fragColor;
#define TAU 6.28318530718
#define MAX_THREADS 10
float glowFn(float x,float str,float dist){return dist/pow(max(x,1e-4),str);}
void main(){
  vec2 uv=gl_FragCoord.xy/iResolution.xy;
  float n=max(uThreadCount,1.);
  float pinchX=uFanMode<.5?.5:(uFanMode<1.5?0.:1.);
  if(uEnableMouse>.5) pinchX=mix(pinchX,uMouse.x,clamp(uMouseStrength,0.,1.)*uMouseActive);
  float spreadDx=uSpread*abs(uv.x-pinchX),baseT=iTime*uSpeed,tauOverN=TAU/n;
  float mirror=uMirror>.5?sign(pinchX-uv.x):1.;
  bool doShimmer=uShimmer>.5;
  float shimmerT=iTime*1.7,invThickness=1./max(uThickness,.01),xFreq=uv.x*uFrequency,yOff=uv.y-uPosition;
  float ciScale=n>1.?1./(n-1.):0.;
  vec3 col=vec3(0.); float gsum=0.;
  for(int idx=0;idx<MAX_THREADS;idx++){
    float i=float(idx); if(i>=n) break;
    float amplitude=spreadDx*(1.+i*uTaper);
    float shimmer=doShimmer?sin(shimmerT+i*1.3)*.35:0.;
    float phase=(baseT+i*tauOverN)*mirror+shimmer;
    float sdf=abs(yOff+sin(xFreq+phase)*amplitude)*invThickness;
    float g=glowFn(sdf,uFalloff,uGlow),ci=i*ciScale;
    col+=g*mix(uColor1,uColor2,ci); gsum+=g;
  }
  col=mix(col,uColor3*gsum,smoothstep(.5,2.2,gsum)*.5);
  float bright=uBrightness;
  if(uEnableMouse>.5){vec2 md=uv-uMouse;bright+=clamp(uMouseStrength,0.,1.)*uMouseActive*exp(-dot(md,md)*6.)*.6;}
  col*=bright; float alpha=clamp(gsum,0.,1.)*uOpacity; vec3 outRgb=col*alpha;
  if(uGrain>.5){float gv=(fract(sin(dot(gl_FragCoord.xy,vec2(12.9898,78.233))+iTime)*43758.5453)-.5)*uGrainIntensity;outRgb=clamp(outRgb+gv,0.,1.);alpha=clamp(alpha+gv,0.,1.);}
  fragColor=vec4(outRgb,alpha);
}`;
const ctxMap = new WeakMap();

export default function WebThreads({
  color1='#5227FF', color2='#FF9FFC', color3='#FFFFFF', speed=.2,
  threadCount=6, frequency=5, spread=.18, taper=1, position=.5,
  fanMode='center', glow=.02, falloff=.6, thickness=1.1, brightness=.6,
  opacity=1, mirror=true, shimmer=false, grain=true, grainIntensity=.05,
  mouseInteraction=true, mouseStrength=.3, className=''
}) {
  const containerRef=useRef(null);
  const mouseRef=useRef({enabled:true,strength:.3});
  useEffect(()=>{
    const container=containerRef.current;
    if(!container) return;
    const renderer=new Renderer({webgl:2,alpha:true,premultipliedAlpha:true,antialias:false,dpr:Math.min(devicePixelRatio||1,2)});
    const gl=renderer.gl; gl.clearColor(0,0,0,0);
    const canvas=gl.canvas;
    Object.assign(canvas.style,{width:'100%',height:'100%',display:'block'});
    container.appendChild(canvas);
    const geometry=new Triangle(gl);
    const program=new Program(gl,{vertex,fragment,uniforms:{
      iTime:{value:0},iResolution:{value:new Float32Array([1,1])},uSpeed:{value:.2},uThreadCount:{value:6},uFrequency:{value:5},uSpread:{value:.18},uTaper:{value:1},uPosition:{value:.5},uFanMode:{value:0},uGlow:{value:.02},uFalloff:{value:.6},uThickness:{value:1.1},uBrightness:{value:.6},uOpacity:{value:1},uMirror:{value:1},uShimmer:{value:0},uGrain:{value:1},uGrainIntensity:{value:.05},uColor1:{value:new Float32Array([1,1,1])},uColor2:{value:new Float32Array([1,1,1])},uColor3:{value:new Float32Array([1,1,1])},uMouse:{value:new Float32Array([.5,.5])},uMouseStrength:{value:.3},uEnableMouse:{value:1},uMouseActive:{value:0}
    }});
    const mesh=new Mesh(gl,{geometry,program}); ctxMap.set(container,{program});
    const setSize=()=>{const r=container.getBoundingClientRect();renderer.setSize(Math.max(1,Math.floor(r.width)),Math.max(1,Math.floor(r.height)));program.uniforms.iResolution.value.set([gl.drawingBufferWidth,gl.drawingBufferHeight]);renderer.render({scene:mesh});};
    const ro=new ResizeObserver(setSize);ro.observe(container);setSize();
    const current=[.5,.5],target=[.5,.5];let active=0,targetActive=0,raf=0,visible=true,pageVisible=!document.hidden;
    const move=e=>{const r=canvas.getBoundingClientRect();target[0]=(e.clientX-r.left)/r.width;target[1]=1-(e.clientY-r.top)/r.height;targetActive=1;};
    const enter=()=>targetActive=1,leave=()=>targetActive=0;
    canvas.addEventListener('mousemove',move);canvas.addEventListener('mouseenter',enter);canvas.addEventListener('mouseleave',leave);
    const t0=performance.now();
    const loop=t=>{program.uniforms.iTime.value=(t-t0)*.001;current[0]+=.05*(target[0]-current[0]);current[1]+=.05*(target[1]-current[1]);active+=.05*(targetActive-active);program.uniforms.uMouse.value.set(current);program.uniforms.uMouseActive.value=active;program.uniforms.uEnableMouse.value=mouseRef.current.enabled?1:0;program.uniforms.uMouseStrength.value=mouseRef.current.strength;renderer.render({scene:mesh});raf=requestAnimationFrame(loop);};
    const start=()=>{if(visible&&pageVisible&&raf===0)raf=requestAnimationFrame(loop);};
    const stop=()=>{if(raf){cancelAnimationFrame(raf);raf=0;}};
    const io=new IntersectionObserver(([entry])=>{visible=entry.isIntersecting;visible?start():stop();});io.observe(container);
    const visibility=()=>{pageVisible=!document.hidden;pageVisible?start():stop();};document.addEventListener('visibilitychange',visibility);start();
    return()=>{stop();ro.disconnect();io.disconnect();document.removeEventListener('visibilitychange',visibility);canvas.removeEventListener('mousemove',move);canvas.removeEventListener('mouseenter',enter);canvas.removeEventListener('mouseleave',leave);ctxMap.delete(container);canvas.remove();gl.getExtension('WEBGL_lose_context')?.loseContext();};
  },[]);
  useEffect(()=>{
    const u=ctxMap.get(containerRef.current)?.program.uniforms;if(!u)return;
    Object.entries({uSpeed:speed,uThreadCount:Math.round(threadCount),uFrequency:frequency,uSpread:spread,uTaper:taper,uPosition:position,uFanMode:FAN_MODE[fanMode]??0,uGlow:glow,uFalloff:falloff,uThickness:thickness,uBrightness:brightness,uOpacity:opacity,uMirror:+mirror,uShimmer:+shimmer,uGrain:+grain,uGrainIntensity:grainIntensity,uMouseStrength:mouseStrength,uEnableMouse:+mouseInteraction}).forEach(([k,v])=>u[k].value=v);
    [[u.uColor1,color1],[u.uColor2,color2],[u.uColor3,color3]].forEach(([uniform,color])=>uniform.value.set(hexToRgb(color)));
    mouseRef.current={enabled:mouseInteraction,strength:mouseStrength};
  },[color1,color2,color3,speed,threadCount,frequency,spread,taper,position,fanMode,glow,falloff,thickness,brightness,opacity,mirror,shimmer,grain,grainIntensity,mouseInteraction,mouseStrength]);
  return <div ref={containerRef} className={`web-threads-container ${className}`.trim()}/>;
}
