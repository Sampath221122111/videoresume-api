import { useState, useEffect, useRef, useCallback } from 'react';
import { supabase, auth, db } from './supabase.js';
import { uploadToCloudinary } from './cloudinary.js';
import { api } from './api.js';

/* ================================================================
   THEME — Monochrome white/black/grey matching 3D robot
   ================================================================ */
const T = {
  bg:'#0b0b0b', surface:'rgba(18,18,18,0.9)', card:'rgba(20,20,20,0.85)',
  elevated:'rgba(28,28,28,0.95)', accent:'#ffffff', accentSoft:'rgba(255,255,255,0.7)',
  gradient:'linear-gradient(135deg,#ffffff,#d4d4d4)',
  text:'#f0f0f0', textSoft:'rgba(200,200,200,0.75)', muted:'rgba(140,140,140,0.6)',
  dim:'rgba(90,90,90,0.5)', success:'#4ade80', warning:'#facc15', danger:'#f87171',
  purple:'#a78bfa', border:'rgba(255,255,255,0.06)', borderHover:'rgba(255,255,255,0.14)',
  glow:'rgba(255,255,255,0.06)', inputBg:'rgba(15,15,15,0.95)',
  font:"'Outfit',system-ui,-apple-system,sans-serif",
};

const STYLES = `
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{overflow-x:hidden;background:${T.bg};font-family:${T.font}}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.06);border-radius:10px}
input::placeholder{color:rgba(90,90,90,0.5)}select{appearance:none}a{text-decoration:none}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes fadeUp{from{transform:translateY(28px);opacity:0}to{transform:translateY(0);opacity:1}}
@keyframes enterUp{from{transform:translateY(25px);opacity:0}to{transform:translateY(0);opacity:1}}
@keyframes cardIn{from{transform:translateY(40px) scale(0.95);opacity:0}to{transform:translateY(0) scale(1);opacity:1}}
@keyframes scaleIn{from{transform:scale(.93);opacity:0}to{transform:scale(1);opacity:1}}
@keyframes pulse{0%,100%{opacity:.3}50%{opacity:1}}
@keyframes slideDown{from{transform:translateY(-16px);opacity:0}to{transform:translateY(0);opacity:1}}
@keyframes glow{0%,100%{box-shadow:0 0 20px rgba(255,255,255,0.03)}50%{box-shadow:0 0 40px rgba(255,255,255,0.08)}}
@keyframes scanLine{0%{top:0}100%{top:100%}}
@keyframes cardReveal{from{transform:translateY(40px) scale(.96);opacity:0}to{transform:translateY(0) scale(1);opacity:1}}
@keyframes cardEnter3D{from{transform:translateY(50px) rotateX(10deg) scale(0.9);opacity:0}to{transform:translateY(0) rotateX(0) scale(1);opacity:1}}
@keyframes skillGlow{0%{transform:translateY(0) scale(1);box-shadow:0 0 0 rgba(255,255,255,0)}50%{transform:translateY(-2px) scale(1.05);box-shadow:0 4px 12px rgba(255,255,255,0.08)}100%{transform:translateY(0) scale(1);box-shadow:0 0 0 rgba(255,255,255,0)}}
@keyframes barShimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}
@keyframes barPulse{0%,100%{opacity:0.4;transform:translateY(-50%) scale(1)}50%{opacity:1;transform:translateY(-50%) scale(1.5)}}
@keyframes optIn{from{opacity:0;transform:translateY(30px) rotateX(12deg)}to{opacity:1;transform:translateY(0) rotateX(0)}}
@keyframes scanMove{0%{top:0;opacity:0}15%{opacity:1}85%{opacity:1}100%{top:100%;opacity:0}}
@keyframes ringPulse{0%{opacity:0.3;transform:scale(0.95)}50%{opacity:0.7;transform:scale(1.05)}100%{opacity:0;transform:scale(1.2)}}
@keyframes camBreathe{0%,100%{transform:scale(1)}50%{transform:scale(1.06)}}
@keyframes recPulse{0%,100%{opacity:1;filter:drop-shadow(0 0 4px #ff4444)}50%{opacity:0.4;filter:drop-shadow(0 0 12px #ff4444)}}
@keyframes recPulseSoft{0%,100%{opacity:0.7}50%{opacity:1}}
@keyframes waveExpand{0%{opacity:0;transform:translateX(-2px)}50%{opacity:0.8}100%{opacity:0;transform:translateX(3px)}}
@keyframes arrowBounce{0%,100%{transform:translateY(0)}50%{transform:translateY(-4px)}}
@keyframes barGrow{from{height:0}}
@keyframes particleDrift{0%{transform:translateY(100vh) translateX(0);opacity:0}10%{opacity:0.15}90%{opacity:0.15}100%{transform:translateY(-20px) translateX(20px);opacity:0}}
@keyframes orbFloat{0%,100%{transform:translate(0,0) scale(1)}25%{transform:translate(30px,-20px) scale(1.05)}50%{transform:translate(-20px,30px) scale(0.95)}75%{transform:translate(15px,15px) scale(1.02)}}
@keyframes orbFadeIn{to{opacity:1}}
@keyframes itemSlide{from{transform:translateX(-30px) translateY(10px);opacity:0}to{transform:translateX(0) translateY(0);opacity:1}}
@keyframes topLine{0%,100%{transform:translateX(-50%);opacity:0}50%{transform:translateX(50%);opacity:1}}
@keyframes pulseDot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.6;transform:scale(0.9)}}
@keyframes navSlide{from{transform:translateX(-12px);opacity:0}to{transform:translateX(0);opacity:1}}
@keyframes logoGlow{0%,100%{box-shadow:0 0 0 rgba(255,255,255,0)}50%{box-shadow:0 0 20px rgba(255,255,255,0.08)}}
`;

/* Helpers */
const DISP=["tempmail.com","throwaway.email","guerrillamail.com","mailinator.com","yopmail.com"];
const isDisp=e=>DISP.includes(e.split("@")[1]?.toLowerCase());
const valPw=p=>{const c={length:p.length>=8,upper:/[A-Z]/.test(p),lower:/[a-z]/.test(p),number:/[0-9]/.test(p),special:/[^A-Za-z0-9]/.test(p)};return{...c,valid:Object.values(c).every(Boolean)}};
const valFile=f=>{if(!f)return{ok:false,err:"No file"};if(!["video/mp4","video/webm","video/quicktime"].includes(f.type))return{ok:false,err:"Only MP4/WebM/MOV"};if(f.size>200*1024*1024)return{ok:false,err:"Max 200MB"};return{ok:true}};
const fmtTime=s=>`${Math.floor(s/60)}:${String(Math.round(s%60)).padStart(2,'0')}`;

/* ================================================================
   PRIMITIVES
   ================================================================ */
const Card=({children,style={},animate=false,delay=0,glow=false,onMouseEnter,onMouseLeave})=>
  <div onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave} style={{background:T.card,border:`1px solid ${T.border}`,borderRadius:16,padding:22,backdropFilter:'blur(24px)',WebkitBackdropFilter:'blur(24px)',transition:'all 0.4s cubic-bezier(0.4,0,0.2,1)',...(animate?{animation:`cardReveal 0.6s ease ${delay}s both`}:{}),...(glow?{boxShadow:`0 0 30px ${T.glow}, inset 0 1px 0 rgba(255,255,255,0.03)`}:{}),...style}}>{children}</div>;

const Input=({label,type="text",value,onChange,placeholder,error,icon,disabled})=>
  <div style={{marginBottom:20}}>
    {label&&<label style={{display:'block',marginBottom:8,fontSize:12,color:T.muted,fontWeight:500,letterSpacing:0.8,textTransform:'uppercase'}}>{label}</label>}
    <div style={{position:'relative'}}>
      {icon&&<span style={{position:'absolute',left:14,top:'50%',transform:'translateY(-50%)',fontSize:14,color:T.dim}}>{icon}</span>}
      <input type={type} value={value} onChange={onChange} placeholder={placeholder} disabled={disabled}
        style={{width:'100%',padding:icon?'14px 16px 14px 42px':'14px 16px',background:T.inputBg,border:`1px solid ${error?'rgba(248,113,113,0.3)':T.border}`,borderRadius:12,color:T.text,fontSize:14,fontFamily:T.font,outline:'none',transition:'all .3s',boxSizing:'border-box'}}
        onFocus={e=>{e.target.style.borderColor='rgba(255,255,255,0.15)';e.target.style.boxShadow='0 0 0 3px rgba(255,255,255,0.03)'}}
        onBlur={e=>{e.target.style.borderColor=error?'rgba(248,113,113,0.3)':T.border;e.target.style.boxShadow='none'}}/>
    </div>
    {error&&<p style={{color:T.danger,fontSize:11.5,marginTop:6}}>{error}</p>}
  </div>;

const Btn=({children,onClick,v="primary",disabled,loading,style:ext={},full=true})=>{
  const[hov,setHov]=useState(false);
  const base={padding:'13px 24px',borderRadius:12,fontSize:14,fontWeight:600,fontFamily:T.font,cursor:disabled?'not-allowed':'pointer',transition:'all .35s',border:'none',display:'inline-flex',alignItems:'center',justifyContent:'center',gap:8,width:full?'100%':'auto',opacity:disabled?0.4:1,transform:hov&&!disabled?'translateY(-1px)':'none'};
  const vs={
    primary:{background:T.gradient,color:'#0b0b0b',boxShadow:hov?'0 8px 32px rgba(255,255,255,0.1)':'0 2px 12px rgba(255,255,255,0.04)'},
    secondary:{background:'rgba(255,255,255,0.04)',color:T.text,border:`1px solid ${T.border}`},
    danger:{background:'rgba(248,113,113,0.06)',color:T.danger,border:'1px solid rgba(248,113,113,0.1)'},
    ghost:{background:'transparent',color:T.accentSoft,padding:'8px 16px'},
    outline:{background:'transparent',color:T.accentSoft,border:'1px solid rgba(255,255,255,0.12)',boxShadow:hov?`0 0 16px ${T.glow}`:'none'},
  };
  return <button onClick={disabled?undefined:onClick} onMouseEnter={()=>setHov(true)} onMouseLeave={()=>setHov(false)} style={{...base,...vs[v],...ext}}>
    {loading&&<span style={{display:'inline-block',width:14,height:14,border:'2px solid rgba(0,0,0,0.15)',borderTopColor:'#000',borderRadius:'50%',animation:'spin .6s linear infinite'}}/>}{children}
  </button>;
};

const Toast=({message,type="info",onClose})=>{useEffect(()=>{const t=setTimeout(onClose,4000);return()=>clearTimeout(t)},[onClose]);const c={success:T.success,error:T.danger,info:'#fff'};return<div style={{position:'fixed',top:20,right:20,zIndex:9999,background:'rgba(12,12,12,0.96)',backdropFilter:'blur(20px)',border:`1px solid ${c[type]}44`,borderRadius:14,padding:'14px 22px',color:c[type],fontSize:13,fontFamily:T.font,animation:'slideDown .35s ease',maxWidth:380,boxShadow:'0 8px 32px rgba(0,0,0,0.5)'}}>{message}</div>};
const LogoutModal=({onOk,onNo})=><div style={{position:'fixed',inset:0,zIndex:999,display:'flex',alignItems:'center',justifyContent:'center',background:'rgba(0,0,0,0.75)',backdropFilter:'blur(12px)'}}><Card style={{maxWidth:380,width:'90%',textAlign:'center',padding:36,animation:'scaleIn .25s ease'}} glow><h3 style={{fontSize:18,fontWeight:700,marginBottom:8}}>Sign Out?</h3><p style={{color:T.muted,fontSize:14,marginBottom:28}}>You'll need to sign in again.</p><div style={{display:'flex',gap:12}}><Btn v="secondary" onClick={onNo}>Cancel</Btn><Btn v="danger" onClick={onOk}>Sign Out</Btn></div></Card></div>;

const AuthWrap=({children,title,sub})=><div style={{minHeight:'100vh',display:'flex',alignItems:'center',justifyContent:'center',padding:24,position:'relative',overflow:'hidden'}}>
  {/* Background video */}
  <video autoPlay muted loop playsInline style={{position:'absolute',inset:0,width:'100%',height:'100%',objectFit:'cover',zIndex:0}}>
    <source src="https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260217_030345_246c0224-10a4-422c-b324-070b7c0eceda.mp4" type="video/mp4"/>
  </video>
  <div style={{position:'absolute',inset:0,background:'rgba(0,0,0,0.7)',zIndex:1}}/>

  {/* Logo top center */}
  <div style={{position:'fixed',top:28,left:'50%',transform:'translateX(-50%)',zIndex:20,display:'flex',alignItems:'center',gap:10}}>
    <div style={{position:'relative',width:36,height:36}}>
      <div style={{position:'absolute',inset:-2,borderRadius:11,background:'conic-gradient(from 0deg, rgba(255,255,255,0.5), rgba(255,255,255,0.05), rgba(255,255,255,0.5))',animation:'logoSpin 6s linear infinite',opacity:0.5}}/>
      <div style={{position:'absolute',inset:0,borderRadius:9,background:'rgba(0,0,0,0.8)',display:'flex',alignItems:'center',justifyContent:'center',backdropFilter:'blur(12px)'}}>
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="23 7 16 12 23 17" fill="rgba(255,255,255,0.1)"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>
      </div>
    </div>
    <span style={{fontSize:18,fontWeight:700,color:'#fff',letterSpacing:-0.5}}>VR<span style={{fontWeight:300,color:'rgba(255,255,255,0.35)'}}>.</span>ai</span>
  </div>

  {/* Glass card */}
  <div style={{position:'relative',zIndex:10,width:'100%',maxWidth:440,padding:36,borderRadius:24,background:'rgba(10,10,12,0.75)',border:'1px solid rgba(255,255,255,0.08)',backdropFilter:'blur(40px)',WebkitBackdropFilter:'blur(40px)',boxShadow:'0 24px 80px rgba(0,0,0,0.5)',animation:'scaleIn .4s ease both',overflow:'hidden'}}>
    <div style={{position:'absolute',top:0,left:'15%',right:'15%',height:1,background:'linear-gradient(90deg,transparent,rgba(255,255,255,0.2),transparent)'}}/>
    <div style={{position:'absolute',top:0,left:0,right:0,height:160,background:'radial-gradient(ellipse at 50% 0%, rgba(255,255,255,0.04), transparent 70%)',pointerEvents:'none'}}/>
    <div style={{textAlign:'center',marginBottom:28,position:'relative'}}>
      <h2 style={{fontSize:24,fontWeight:800,color:'#fff',marginBottom:4,letterSpacing:-0.5}}>{title}</h2>
      {sub&&<p style={{fontSize:13.5,color:T.muted}}>{sub}</p>}
    </div>
    <div style={{position:'relative'}}>{children}</div>
  </div>
</div>;

/* Auth Pages */
const SignupPage=({go,show})=>{const[e,sE]=useState(''),[p,sP]=useState(''),[cp,sCP]=useState(''),[er,sEr]=useState({}),[ld,sLd]=useState(false);const run=async()=>{const x={};if(!e)x.e="Required";else if(!/\S+@\S+\.\S+/.test(e))x.e="Invalid";else if(isDisp(e))x.e="Blocked";if(!valPw(p).valid)x.p="8+ chars: upper, lower, number, special";if(p!==cp)x.cp="Don't match";sEr(x);if(Object.keys(x).length)return;sLd(true);try{const result=await auth.signUp(e,p);
  /* Supabase doesn't throw error for duplicate emails — it returns a fake user with
     identities: [] (empty array). Real new users have identities with data. */
  if(result?.user&&result.user.identities&&result.user.identities.length===0){
    sEr({e:"This email is already registered. Please sign in instead."});sLd(false);return;
  }
  /* Also check if user already has confirmed_at set (another sign of existing account) */
  if(result?.user?.confirmed_at&&!result?.user?.email_confirmed_at){
    sEr({e:"This email is already registered. Please sign in instead."});sLd(false);return;
  }
  show("Account created! Check your email to verify.","success");go("login")
  }catch(err){const msg=(err.message||"").toLowerCase();if(msg.includes("already registered")||msg.includes("already been registered")||msg.includes("user already")||msg.includes("email already")||msg.includes("duplicate")||msg.includes("unique constraint")||msg.includes("already exists")){sEr({e:"This email is already registered. Please sign in instead."})}else if(msg.includes("password")){sEr({p:err.message})}else if(msg.includes("rate")||msg.includes("too many")){sEr({e:"Too many attempts. Please wait a minute and try again."})}else{sEr({e:err.message})}}finally{sLd(false)}};return<AuthWrap title="Create Account" sub="Start building your AI resume"><Input label="Email" type="email" value={e} onChange={v=>sE(v.target.value)} placeholder="you@email.com" error={er.e} icon="✉"/><Input label="Password" type="password" value={p} onChange={v=>sP(v.target.value)} placeholder="8+ characters" error={er.p} icon="🔒"/><Input label="Confirm" type="password" value={cp} onChange={v=>sCP(v.target.value)} placeholder="Re-enter" error={er.cp} icon="🔒"/>{p&&<div style={{marginBottom:16,display:'flex',flexWrap:'wrap',gap:5}}>{Object.entries(valPw(p)).filter(([k])=>k!=="valid").map(([k,v])=><span key={k} style={{fontSize:10.5,padding:'3px 10px',borderRadius:20,background:v?'rgba(74,222,128,0.06)':'rgba(248,113,113,0.06)',color:v?T.success:'rgba(248,113,113,0.6)'}}>{v?"✓":"✗"} {k}</span>)}</div>}<Btn onClick={run} loading={ld} disabled={ld}>Create Account</Btn><p style={{textAlign:'center',marginTop:18,fontSize:13,color:T.dim}}>Have an account? <span onClick={()=>go("login")} style={{color:'#fff',cursor:'pointer',fontWeight:600}}>Sign In</span></p></AuthWrap>};
const LoginPage=({go,show})=>{const[e,sE]=useState(''),[p,sP]=useState(''),[er,sEr]=useState({}),[ld,sLd]=useState(false);const run=async()=>{const x={};if(!e)x.e="Required";else if(!/\S+@\S+\.\S+/.test(e))x.e="Invalid email";if(!p)x.p="Required";sEr(x);if(Object.keys(x).length)return;sLd(true);try{await auth.signIn(e,p);show("Welcome back!","success")}catch(err){const msg=(err.message||"").toLowerCase();if(msg.includes("invalid login credentials")||msg.includes("invalid")){sEr({g:"Invalid email or password. Please check your credentials and try again."})}else if(msg.includes("email not confirmed")){sEr({g:"Please verify your email first. Check your inbox for the confirmation link."})}else if(msg.includes("too many requests")||msg.includes("rate")){sEr({g:"Too many login attempts. Please wait a few minutes and try again."})}else{sEr({g:err.message||"Login failed. Please try again."})}}finally{sLd(false)}};return<AuthWrap title="Welcome Back" sub="Sign in to your dashboard">{er.g&&<div style={{background:'rgba(248,113,113,0.05)',border:'1px solid rgba(248,113,113,0.08)',borderRadius:10,padding:'11px 14px',marginBottom:16,color:T.danger,fontSize:12.5}}>{er.g}</div>}<Input label="Email" type="email" value={e} onChange={v=>sE(v.target.value)} placeholder="you@email.com" error={er.e} icon="✉"/><Input label="Password" type="password" value={p} onChange={v=>sP(v.target.value)} placeholder="Password" error={er.p} icon="🔒"/><div style={{textAlign:'right',marginTop:-12,marginBottom:16}}><span onClick={()=>go("forgot")} style={{color:T.muted,fontSize:12.5,cursor:'pointer'}}>Forgot?</span></div><Btn onClick={run} loading={ld} disabled={ld}>Sign In</Btn><p style={{textAlign:'center',marginTop:18,fontSize:13,color:T.dim}}>No account? <span onClick={()=>go("signup")} style={{color:'#fff',cursor:'pointer',fontWeight:600}}>Sign Up</span></p></AuthWrap>};
const ForgotPage=({go,show})=>{const[e,sE]=useState(''),[ld,sLd]=useState(false),[sent,sSent]=useState(false);const run=async()=>{if(!e)return;sLd(true);try{await auth.resetPassword(e);sSent(true);show("Link sent!","success")}catch(x){show(x.message,"error")}finally{sLd(false)}};return<AuthWrap title="Reset Password">{!sent&&<><Input label="Email" type="email" value={e} onChange={v=>sE(v.target.value)} placeholder="you@email.com" icon="✉"/><Btn onClick={run} loading={ld}>Send Link</Btn></>}{sent&&<Btn onClick={()=>go("login")}>Back</Btn>}<p style={{textAlign:'center',marginTop:16}}><span onClick={()=>go("login")} style={{color:T.muted,cursor:'pointer',fontSize:13}}>← Back</span></p></AuthWrap>};
const ProfileSetup=({go,show,session,setProfile,loadSubs})=>{const[n,sN]=useState(''),[u,sU]=useState(''),[b,sB]=useState(''),[y,sY]=useState(''),[er,sEr]=useState({}),[ld,sLd]=useState(false);const run=async()=>{const x={};if(!n.trim())x.n="Required";if(!u.trim())x.u="Required";if(!b.trim())x.b="Required";if(!y)x.y="Required";sEr(x);if(Object.keys(x).length)return;sLd(true);try{const p=await db.updateProfile(session.user.id,{full_name:n.trim(),university:u.trim(),branch:b.trim(),year_of_study:parseInt(y)});setProfile(p);await loadSubs(session.user.id);show("Saved!","success");go("dashboard")}catch(err){show(err.message,"error")}finally{sLd(false)}};return<AuthWrap title="Complete Profile" sub="Tell us about yourself"><Input label="Full Name" value={n} onChange={v=>sN(v.target.value)} placeholder="Your name" error={er.n} icon="👤"/><Input label="University" value={u} onChange={v=>sU(v.target.value)} placeholder="e.g. AEC Engineering" error={er.u} icon="🎓"/><Input label="Branch" value={b} onChange={v=>sB(v.target.value)} placeholder="e.g. CSE" error={er.b} icon="📚"/><div style={{marginBottom:20}}><label style={{display:'block',marginBottom:8,fontSize:12,color:T.muted,fontWeight:500,letterSpacing:0.8,textTransform:'uppercase'}}>Year</label><select value={y} onChange={v=>sY(v.target.value)} style={{width:'100%',padding:'14px 16px',background:T.inputBg,border:`1px solid ${T.border}`,borderRadius:12,color:y?T.text:T.dim,fontSize:14,fontFamily:T.font,outline:'none'}}><option value="">Select</option>{[1,2,3,4].map(i=><option key={i} value={i}>{i}{['st','nd','rd','th'][i-1]} Year</option>)}</select></div><Btn onClick={run} loading={ld} disabled={ld}>Save & Continue →</Btn></AuthWrap>};

/* Face Scan */
const FaceScan=({stage,prog=0})=>{
  const stages=['Downloading','Extracting audio','Transcribing','Analyzing voice & face','Building resume & clip','Uploading results','Saving'];
  const maxProg=useRef(0);
  const maxIdx=useRef(0);

  // Never let progress go backward
  if(prog>maxProg.current) maxProg.current=prog;
  const displayProg=maxProg.current;

  // Match current stage text to stage list — flexible matching
  const stageL=(stage||'').toLowerCase();
  let matchIdx=-1;
  if(stageL.includes('download'))matchIdx=0;
  else if(stageL.includes('extract'))matchIdx=1;
  else if(stageL.includes('transcrib'))matchIdx=2;
  else if(stageL.includes('analy')||stageL.includes('voice')||stageL.includes('face'))matchIdx=3;
  else if(stageL.includes('resum')||stageL.includes('clip')||stageL.includes('build')||stageL.includes('generat'))matchIdx=4;
  else if(stageL.includes('upload'))matchIdx=5;
  else if(stageL.includes('sav')||stageL.includes('complet')||stageL.includes('done'))matchIdx=6;

  // Fallback to progress-based index
  const calcIdx=matchIdx>=0?matchIdx:Math.min(Math.floor(displayProg/15),stages.length-1);

  // Never go backward on stage index either
  if(calcIdx>maxIdx.current) maxIdx.current=calcIdx;
  const activeIdx=maxIdx.current;

  return <div style={{padding:'28px 0'}}>
    {/* Animated progress ring */}
    <div style={{display:'flex',justifyContent:'center',marginBottom:28}}>
      <div style={{position:'relative',width:100,height:100}}>
        <svg width="100" height="100" viewBox="0 0 100 100" style={{transform:'rotate(-90deg)'}}>
          <circle cx="50" cy="50" r="42" fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="4"/>
          <circle cx="50" cy="50" r="42" fill="none" stroke="#fff" strokeWidth="4" strokeLinecap="round" strokeDasharray={264} strokeDashoffset={264-(264*Math.min(displayProg,100)/100)} style={{transition:'stroke-dashoffset 0.8s ease'}}/>
        </svg>
        <div style={{position:'absolute',inset:0,display:'flex',alignItems:'center',justifyContent:'center',flexDirection:'column'}}>
          <div style={{fontSize:22,fontWeight:800,color:'#fff',fontFamily:"'JetBrains Mono',monospace",letterSpacing:-1}}>{Math.round(displayProg)}%</div>
        </div>
        <div style={{position:'absolute',top:-3,left:'50%',marginLeft:-3,width:6,height:6,borderRadius:'50%',background:'#fff',boxShadow:'0 0 10px rgba(255,255,255,0.8)',animation:'spin 3s linear infinite',transformOrigin:'3px 53px'}}/>
      </div>
    </div>

    <div style={{textAlign:'center',marginBottom:24}}>
      <div style={{fontSize:15,fontWeight:600,color:'#fff',marginBottom:4}}>{stage||'Processing...'}</div>
      <div style={{fontSize:12,color:T.dim}}>This usually takes 1-2 minutes</div>
    </div>

    <div style={{display:'flex',flexDirection:'column',gap:4}}>
      {stages.map((s,i)=>{
        const done=i<activeIdx;
        const active=i===activeIdx;
        return <div key={s} style={{display:'flex',alignItems:'center',gap:10,padding:'7px 12px',borderRadius:8,background:active?'rgba(255,255,255,0.03)':'transparent',transition:'all 0.3s'}}>
          <div style={{width:20,height:20,borderRadius:6,display:'flex',alignItems:'center',justifyContent:'center',flexShrink:0,background:done?'rgba(74,222,128,0.1)':active?'rgba(255,255,255,0.06)':'transparent',border:`1px solid ${done?'rgba(74,222,128,0.2)':active?'rgba(255,255,255,0.1)':'rgba(255,255,255,0.04)'}`,transition:'all 0.3s'}}>
            {done?<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#4ade80" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
            :active?<div style={{width:6,height:6,borderRadius:'50%',background:'#fff',animation:'pulse 1.5s ease infinite'}}/>
            :<div style={{width:4,height:4,borderRadius:'50%',background:'rgba(255,255,255,0.15)'}}/>}
          </div>
          <span style={{fontSize:12,fontWeight:active?600:400,color:done?T.success:active?'#fff':'rgba(255,255,255,0.2)',transition:'all 0.3s'}}>{s}</span>
          {active&&<div style={{marginLeft:'auto',width:16,height:16,border:'2px solid rgba(255,255,255,0.1)',borderTopColor:'#fff',borderRadius:'50%',animation:'spin 0.8s linear infinite'}}/>}
        </div>;
      })}
    </div>
  </div>;
};

/* Recorder */
const Recorder=({onDone,onCancel})=>{const vr=useRef(),mr=useRef(),sr=useRef(),ch=useRef([]),tr=useRef();const[rec,sR]=useState(false),[cd,sC]=useState(null),[el,sE]=useState(0),[rdy,sRdy]=useState(false),[err,sErr]=useState(null),[pv,sPv]=useState(null);const start=useCallback(async()=>{try{const s=await navigator.mediaDevices.getUserMedia({video:{width:1280,height:720,facingMode:"user"},audio:true});sr.current=s;if(vr.current)vr.current.srcObject=s;sRdy(true)}catch{sErr("Camera denied.")}},[]);useEffect(()=>{start();return()=>{sr.current?.getTracks().forEach(t=>t.stop());clearInterval(tr.current)}},[start]);const scd=()=>{sC(3);let c=3;const iv=setInterval(()=>{c--;if(c<=0){clearInterval(iv);sC(null);br()}else sC(c)},1000)};const br=()=>{ch.current=[];const mime=MediaRecorder.isTypeSupported("video/webm;codecs=vp9,opus")?"video/webm;codecs=vp9,opus":"video/webm";const m=new MediaRecorder(sr.current,{mimeType:mime});m.ondataavailable=e=>{if(e.data.size>0)ch.current.push(e.data)};m.onstop=()=>{sPv(new Blob(ch.current,{type:"video/webm"}));sr.current?.getTracks().forEach(t=>t.stop())};mr.current=m;m.start(100);sR(true);sE(0);tr.current=setInterval(()=>sE(p=>{if(p>=300){sp();return 300}return p+1}),1000)};const sp=()=>{clearInterval(tr.current);mr.current?.stop();sR(false)};const fm=s=>`${Math.floor(s/60).toString().padStart(2,"0")}:${(s%60).toString().padStart(2,"0")}`;if(err)return<Card style={{textAlign:'center',padding:40}}><p style={{color:T.danger,marginBottom:20}}>{err}</p><Btn v="secondary" onClick={onCancel}>Back</Btn></Card>;return<div><div style={{borderRadius:14,overflow:'hidden',background:'#000',marginBottom:16,position:'relative',aspectRatio:'16/9'}}>{pv?<video src={URL.createObjectURL(pv)} controls style={{width:'100%',height:'100%',objectFit:'cover',display:'block'}}/>:<video ref={vr} autoPlay muted playsInline style={{width:'100%',height:'100%',objectFit:'cover',display:'block',transform:'scaleX(-1)'}}/>}{cd!==null&&<div style={{position:'absolute',inset:0,display:'flex',alignItems:'center',justifyContent:'center',background:'rgba(0,0,0,0.6)'}}><span style={{fontSize:64,fontWeight:800,color:'#fff'}}>{cd}</span></div>}{rec&&<div style={{position:'absolute',top:12,left:12,display:'flex',alignItems:'center',gap:6,background:'rgba(0,0,0,0.7)',borderRadius:20,padding:'5px 14px'}}><div style={{width:8,height:8,borderRadius:'50%',background:T.danger,animation:'pulse 1s ease infinite'}}/><span style={{color:'#fff',fontSize:12,fontWeight:600}}>REC {fm(el)}</span></div>}{rec&&<div style={{position:'absolute',bottom:0,left:0,right:0,height:3,background:'rgba(255,255,255,0.06)'}}><div style={{height:'100%',background:T.gradient,width:`${(el/300)*100}%`,transition:'width 1s'}}/></div>}</div>{pv?<div style={{display:'flex',gap:10}}><Btn v="secondary" onClick={()=>{sPv(null);sE(0);start()}}>Retake</Btn><Btn onClick={()=>onDone(new File([pv],`rec_${Date.now()}.webm`,{type:"video/webm"}))}>Use This</Btn></div>:rec?<Btn v="danger" onClick={sp}>Stop</Btn>:rdy?<div style={{display:'flex',gap:10}}><Btn v="secondary" onClick={onCancel} full={false}>Cancel</Btn><Btn onClick={scd}>Start Recording</Btn></div>:<div style={{textAlign:'center',padding:20}}><div style={{width:20,height:20,border:`2px solid ${T.border}`,borderTopColor:'#fff',borderRadius:'50%',animation:'spin .8s linear infinite',margin:'0 auto'}}/></div>}</div>};

/* ================================================================
   VISME FORM PAGE — Contact/Credential Form
   ================================================================ */
const VismeFormPage=({go})=>{
  return <div style={{position:'relative',minHeight:'100vh',width:'100%',background:'#000',fontFamily:T.font,overflow:'hidden'}}>
    
    {/* Background video — same as landing page */}
    <video autoPlay muted loop playsInline style={{position:'absolute',inset:0,width:'100%',height:'100%',objectFit:'cover',zIndex:0}}>
      <source src="https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260217_030345_246c0224-10a4-422c-b324-070b7c0eceda.mp4" type="video/mp4"/>
    </video>
    <div style={{position:'absolute',inset:0,background:'rgba(0,0,0,0.75)',zIndex:1}}/>

    {/* Back button */}
    <button onClick={()=>go("landing")} style={{
      position:'fixed',top:24,left:24,zIndex:50,
      display:'flex',alignItems:'center',gap:8,padding:'10px 20px',borderRadius:12,
      background:'rgba(0,0,0,0.5)',border:'1px solid rgba(255,255,255,0.1)',
      color:'rgba(255,255,255,0.7)',fontSize:13,fontWeight:500,cursor:'pointer',
      fontFamily:T.font,backdropFilter:'blur(16px)',WebkitBackdropFilter:'blur(16px)',
      transition:'all 0.3s',
    }} onMouseEnter={e=>{e.currentTarget.style.background='rgba(0,0,0,0.7)';e.currentTarget.style.color='#fff';e.currentTarget.style.borderColor='rgba(255,255,255,0.2)'}}
       onMouseLeave={e=>{e.currentTarget.style.background='rgba(0,0,0,0.5)';e.currentTarget.style.color='rgba(255,255,255,0.7)';e.currentTarget.style.borderColor='rgba(255,255,255,0.1)'}}>
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/></svg>
      Back
    </button>

    {/* Logo top-center */}
    <div style={{position:'fixed',top:24,left:'50%',transform:'translateX(-50%)',zIndex:50,display:'flex',alignItems:'center',gap:10,opacity:0.6}}>
      <div style={{width:32,height:32,borderRadius:8,background:'rgba(0,0,0,0.5)',border:'1px solid rgba(255,255,255,0.08)',display:'flex',alignItems:'center',justifyContent:'center',backdropFilter:'blur(12px)'}}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="23 7 16 12 23 17" fill="rgba(255,255,255,0.1)"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>
      </div>
      <span style={{fontSize:16,fontWeight:700,color:'#fff',letterSpacing:-0.5}}>VR<span style={{fontWeight:300,color:'rgba(255,255,255,0.4)'}}>.</span>ai</span>
    </div>

    {/* Visme Form — full page iframe */}
    <iframe 
      src="https://forms.visme.co/formsPlayer/op6o1j09-untitled-project"
      style={{
        position:'relative',zIndex:10,
        width:'100%',height:'100vh',border:'none',
        background:'transparent',
      }}
      title="VideoResumeAI Registration Form"
      allow="camera;microphone"
    />
  </div>;
};

/* ================================================================
   LANDING PAGE
   ================================================================ */
const LandingPage=({go})=>{
  const[vis,setVis]=useState(false);
  useEffect(()=>{setTimeout(()=>setVis(true),200)},[]);
  const VIDEO_URL="https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260217_030345_246c0224-10a4-422c-b324-070b7c0eceda.mp4";

  const pills=[
    {label:'Speech Analysis',icon:'M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3zM19 10v2a7 7 0 01-14 0v-2M12 19v4'},
    {label:'Face Detection',icon:'M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2M12 3a5 5 0 100 10 5 5 0 000-10z'},
    {label:'PDF Resume',icon:'M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8zM14 2v6h6M16 13H8M16 17H8'},
    {label:'Highlight Clip',icon:'M23 7l-7 5 7 5zM1 5h15v14H1z'},
  ];

  return <div style={{position:'relative',minHeight:'100vh',width:'100%',background:'#000',overflow:'hidden',display:'flex',flexDirection:'column',fontFamily:"'General Sans', system-ui, sans-serif"}}>
    <style>{`
      @import url('https://api.fontshare.com/v2/css?f[]=general-sans@300,400,500,600,700&display=swap');
      @keyframes heroFade{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
      @keyframes logoSpin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
      @keyframes logoGlowPulse{0%,100%{opacity:0.4}50%{opacity:0.8}}
      @keyframes scanLine{0%,100%{left:0;opacity:0}50%{left:calc(100% - 30px);opacity:1}}
      @keyframes pillFade{from{opacity:0;transform:translateY(12px) scale(0.95)}to{opacity:1;transform:translateY(0) scale(1)}}
    `}</style>

    {/* Background video */}
    <video autoPlay muted loop playsInline style={{position:'absolute',inset:0,width:'100%',height:'100%',objectFit:'cover',zIndex:0}}>
      <source src={VIDEO_URL} type="video/mp4"/>
    </video>
    <div style={{position:'absolute',inset:0,background:'rgba(0,0,0,0.55)',zIndex:1}}/>

    {/* Logo top-left */}
    <nav style={{position:'relative',zIndex:10,display:'flex',justifyContent:'flex-start',alignItems:'center',padding:'24px 36px',width:'100%',opacity:vis?1:0,transition:'opacity .8s ease'}}>
      <div style={{display:'flex',alignItems:'center',gap:12}}>
        <div style={{position:'relative',width:40,height:40}}>
          <div style={{position:'absolute',inset:-2,borderRadius:12,background:'conic-gradient(from 0deg, rgba(255,255,255,0.5), rgba(255,255,255,0.05), rgba(255,255,255,0.5))',animation:'logoSpin 6s linear infinite',opacity:0.6}}/>
          <div style={{position:'absolute',inset:0,borderRadius:10,background:'#000',display:'flex',alignItems:'center',justifyContent:'center',boxShadow:'0 0 30px rgba(255,255,255,0.06)'}}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="23 7 16 12 23 17" fill="rgba(255,255,255,0.1)"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>
          </div>
          <div style={{position:'absolute',inset:-6,borderRadius:16,background:'radial-gradient(circle, rgba(255,255,255,0.08), transparent 70%)',animation:'logoGlowPulse 3s ease-in-out infinite',pointerEvents:'none'}}/>
        </div>
        <div style={{display:'flex',flexDirection:'column',lineHeight:1}}>
          <span style={{fontSize:20,fontWeight:700,color:'#fff',letterSpacing:-1}}>VR<span style={{fontWeight:300,color:'rgba(255,255,255,0.4)'}}>.</span>ai</span>
          <span style={{fontSize:7.5,fontWeight:500,color:'rgba(255,255,255,0.2)',letterSpacing:2.5,textTransform:'uppercase',marginTop:2}}>Video Resume</span>
        </div>
      </div>
    </nav>

    {/* Hero — pushed lower with flex-end + generous bottom padding */}
    <div style={{position:'relative',zIndex:10,flex:1,display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',textAlign:'center',padding:'60px 24px 100px',gap:0}}>

      {/* Heading */}
      <h1 style={{fontSize:'clamp(42px, 7vw, 72px)',fontWeight:600,lineHeight:1.08,maxWidth:700,letterSpacing:-2.5,margin:0,background:'linear-gradient(144.5deg, #ffffff 28%, rgba(255,255,255,0.2) 100%)',WebkitBackgroundClip:'text',backgroundClip:'text',WebkitTextFillColor:'transparent',color:'transparent',animation:'heroFade 1s cubic-bezier(0.23,1,0.32,1) 0.2s both'}}>Your Voice.<br/>Your Resume.</h1>

      {/* Caption */}
      <p style={{fontSize:15,fontWeight:400,color:'rgba(255,255,255,0.4)',maxWidth:360,lineHeight:1.6,marginTop:24,marginBottom:40,animation:'heroFade 1s cubic-bezier(0.23,1,0.32,1) 0.4s both'}}>
        Record yourself. Let AI build your resume.
      </p>

      {/* Two CTA Buttons */}
      <div style={{display:'flex',gap:14,alignItems:'center',marginBottom:56,animation:'heroFade 1s cubic-bezier(0.23,1,0.32,1) 0.6s both'}}>
        <button onClick={()=>go("login")} style={{padding:'13px 32px',borderRadius:999,fontSize:14,fontWeight:500,background:'rgba(255,255,255,0.06)',color:'rgba(255,255,255,0.8)',border:'1px solid rgba(255,255,255,0.1)',cursor:'pointer',fontFamily:'inherit',backdropFilter:'blur(12px)',WebkitBackdropFilter:'blur(12px)',transition:'all 0.35s cubic-bezier(0.23,1,0.32,1)'}} onMouseEnter={e=>{e.target.style.background='rgba(255,255,255,0.1)';e.target.style.borderColor='rgba(255,255,255,0.2)';e.target.style.color='#fff';e.target.style.transform='translateY(-2px)'}} onMouseLeave={e=>{e.target.style.background='rgba(255,255,255,0.06)';e.target.style.borderColor='rgba(255,255,255,0.1)';e.target.style.color='rgba(255,255,255,0.8)';e.target.style.transform='none'}}>Sign In</button>
        <div onClick={()=>go("signup")} style={{position:'relative',padding:0.6,borderRadius:999,background:'#fff',display:'inline-block',cursor:'pointer',transition:'transform 0.35s cubic-bezier(0.23,1,0.32,1)'}} onMouseEnter={e=>e.currentTarget.style.transform='translateY(-2px) scale(1.03)'} onMouseLeave={e=>e.currentTarget.style.transform='none'}>
          <div style={{position:'relative',background:'#fff',borderRadius:999,padding:'13px 32px',fontSize:14,fontWeight:600,color:'#000',overflow:'hidden',display:'flex',alignItems:'center',gap:8}}>
            <div style={{position:'absolute',top:-6,left:'15%',right:'15%',height:14,background:'radial-gradient(ellipse at center, rgba(255,255,255,0.9), transparent 70%)',filter:'blur(4px)',pointerEvents:'none'}}/>
            Get Started
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
          </div>
        </div>
      </div>

      {/* Feature pills row */}
      <div style={{display:'flex',gap:12,justifyContent:'center',flexWrap:'wrap'}}>
        {pills.map((p,i)=>{
          const[h,setH]=useState(false);
          return <div key={p.label} onMouseEnter={()=>setH(true)} onMouseLeave={()=>setH(false)} style={{
            display:'flex',alignItems:'center',gap:10,padding:'10px 20px 10px 14px',borderRadius:999,
            background:h?'rgba(255,255,255,0.08)':'rgba(255,255,255,0.04)',
            border:`1px solid ${h?'rgba(255,255,255,0.15)':'rgba(255,255,255,0.06)'}`,
            backdropFilter:'blur(16px)',WebkitBackdropFilter:'blur(16px)',
            transition:'all 0.35s cubic-bezier(0.23,1,0.32,1)',cursor:'default',
            transform:h?'translateY(-3px)':'none',
            boxShadow:h?'0 8px 24px rgba(0,0,0,0.3)':'none',
            animation:`pillFade 0.6s cubic-bezier(0.34,1.56,0.64,1) ${0.9+i*0.1}s both`,
          }}>
            <div style={{width:28,height:28,borderRadius:8,background:'rgba(255,255,255,0.06)',display:'flex',alignItems:'center',justifyContent:'center',transition:'transform 0.4s',transform:h?'scale(1.15) rotate(-5deg)':'none'}}>
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d={p.icon}/></svg>
            </div>
            <span style={{fontSize:12.5,fontWeight:500,color:h?'#fff':'rgba(255,255,255,0.6)',transition:'color 0.3s'}}>{p.label}</span>
          </div>;
        })}
      </div>
    </div>
  </div>;
};


/* ================================================================
   MAIN APP
   ================================================================ */
export default function App(){
  const[page,setPage]=useState("loading");const[session,setSession]=useState(null);const[profile,setProfile]=useState(null);const[toast,setToast]=useState(null);const[showLogout,setShowLogout]=useState(false);const[subs,setSubs]=useState([]);const[tab,setTab]=useState("overview");const[selSub,setSelSub]=useState(null);const pr=useRef("loading");const hi=useRef(false);
  const go=p=>{pr.current=p;setPage(p)};const show=(m,t="info")=>setToast({message:m,type:t});
  useEffect(()=>{let alive=true;auth.getSession().then(s=>{if(!alive)return;setSession(s);if(s?.user)lp(s.user.id,true);else go("landing");hi.current=true});const{data:{subscription}}=auth.onAuthStateChange((ev,s)=>{if(!alive)return;setSession(s);if(ev==='SIGNED_OUT'){setProfile(null);setSubs([]);go("landing")}else if(ev==='PASSWORD_RECOVERY'){go("reset-password")}else if(ev==='SIGNED_IN'&&hi.current&&s?.user)lp(s.user.id,true)});return()=>{alive=false;subscription.unsubscribe()}},[]);
  const lp=async(uid,init=false)=>{try{const p=await db.getProfile(uid);setProfile(p);await ls(uid);if(init&&["loading","landing","login","signup"].includes(pr.current))go(p?.full_name?"dashboard":"profile")}catch{if(init&&pr.current==="loading")go("profile")}};
  const ls=async uid=>{try{const s=await db.getSubmissions(uid);setSubs(s)}catch(e){console.error(e)}};
  const hl=async()=>{setShowLogout(false);await auth.signOut();go("landing");show("Signed out","success")};

  return<div style={{minHeight:'100vh',color:T.text,fontFamily:T.font}}><style>{STYLES}</style>
    {toast&&<Toast message={toast.message} type={toast.type} onClose={()=>setToast(null)}/>}
    {showLogout&&<LogoutModal onOk={hl} onNo={()=>setShowLogout(false)}/>}
    {page==="loading"&&<div style={{display:'flex',alignItems:'center',justifyContent:'center',minHeight:'100vh',background:T.bg}}><div style={{width:32,height:32,border:`3px solid ${T.border}`,borderTopColor:'#fff',borderRadius:'50%',animation:'spin .8s linear infinite'}}/></div>}
    {page==="landing"&&<LandingPage go={go}/>}
    {page==="form"&&<VismeFormPage go={go}/>}
    {page==="login"&&<LoginPage go={go} show={show}/>}
    {page==="signup"&&<SignupPage go={go} show={show}/>}
    {page==="forgot"&&<ForgotPage go={go} show={show}/>}
    {page==="reset-password"&&<ResetPasswordPage go={go} show={show}/>}
    {page==="profile"&&<ProfileSetup go={go} show={show} session={session} setProfile={setProfile} loadSubs={ls}/>}
    {page==="dashboard"&&<Dash go={go} show={show} session={session} profile={profile} setProfile={setProfile} subs={subs} onLogout={()=>setShowLogout(true)} ls={ls} tab={tab} setTab={setTab} selSub={selSub} setSelSub={setSelSub}/>}
    {page==="upload"&&<UploadPg go={go} show={show} session={session} profile={profile} ls={ls}/>}
  </div>;
}

/* Sidebar Icon — supports multi-path SVGs */
const SI=({paths})=><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">{paths.map((p,i)=><path key={i} d={p}/>)}</svg>;

const navI=[
  {id:'overview',l:'Overview',paths:['M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z','M9 22V12h6v10']},
  {id:'submissions',l:'Submissions',paths:['M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z','M14 2v6h6']},
  {id:'resumes',l:'Resumes',paths:['M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z','M14 2v6h6','M16 13H8','M16 17H8']},
  {id:'clips',l:'Clips',paths:['M23 7l-7 5 7 5z','M1 5h15v14H1z']},
  {id:'analytics',l:'Analytics',paths:['M18 20V10','M12 20V4','M6 20v-6']},
];

/* Nav Item Component with full animations */
const NavItem=({n,active,onClick,badge,delay=0})=>{
  const[hov,setHov]=useState(false);
  return <button onClick={onClick} onMouseEnter={()=>setHov(true)} onMouseLeave={()=>setHov(false)} style={{
    display:'flex',alignItems:'center',gap:12,padding:'12px 14px',
    borderRadius:11,fontSize:13.5,fontWeight:active?700:500,
    color:active?'#fff':(hov?'rgba(255,255,255,0.9)':T.muted),
    background:active?'linear-gradient(90deg,rgba(255,255,255,0.07),rgba(255,255,255,0.02))':(hov?'rgba(255,255,255,0.02)':'transparent'),
    border:`1px solid ${active?'rgba(255,255,255,0.08)':'transparent'}`,
    cursor:'pointer',fontFamily:T.font,width:'100%',textAlign:'left',
    transition:'all 0.3s cubic-bezier(0.23,1,0.32,1)',
    position:'relative',overflow:'hidden',
    paddingLeft:hov&&!active?18:14,
    boxShadow:active?'inset 0 1px 0 rgba(255,255,255,0.05), 0 4px 20px rgba(0,0,0,0.3)':'none',
    animation:`navSlide 0.4s ease ${delay}s both`,
  }}>
    {/* Slide indicator bar */}
    <div style={{position:'absolute',left:0,top:'50%',transform:'translateY(-50%)',width:3,height:active?22:(hov?16:0),borderRadius:'0 4px 4px 0',background:'#fff',boxShadow:'0 0 8px rgba(255,255,255,0.4)',transition:'height 0.3s cubic-bezier(0.34,1.56,0.64,1)'}}/>
    {/* Shine sweep */}
    <div style={{position:'absolute',inset:0,background:'linear-gradient(90deg,transparent,rgba(255,255,255,0.04) 50%,transparent)',transform:hov?'translateX(100%)':'translateX(-100%)',transition:'transform 0.6s',pointerEvents:'none'}}/>
    <div style={{transform:hov?'scale(1.1)':'scale(1)',transition:'transform 0.4s',filter:active?'drop-shadow(0 0 6px rgba(255,255,255,0.3))':'none'}}><SI paths={n.paths}/></div>
    <span style={{position:'relative',zIndex:1}}>{n.l}</span>
    {badge&&<span style={{marginLeft:'auto',padding:'2px 8px',borderRadius:6,fontSize:9,fontWeight:700,letterSpacing:0.5,background:'rgba(255,255,255,0.06)',color:'rgba(255,255,255,0.6)',fontFamily:"'JetBrains Mono',monospace",textTransform:'uppercase',position:'relative',zIndex:1}}>{badge}</span>}
  </button>;
};

/* Dashboard */
const Dash=({go,show,session,profile,setProfile,subs,onLogout,ls,tab,setTab,selSub,setSelSub})=>{
  const comp=subs.filter(s=>s.status==="completed");
  useEffect(()=>{if(!session?.user?.id)return;const iv=setInterval(()=>ls(session.user.id),10000);return()=>clearInterval(iv)},[session?.user?.id]);
  const accountNav=[
    {id:'prof',l:'Profile',paths:['M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2','M12 7a4 4 0 100 8 4 4 0 000-8z']},
    {id:'sett',l:'Settings',paths:['M12 15a3 3 0 100-6 3 3 0 000 6z','M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 01-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 11-2.83-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 112.83-2.83l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 112.83 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z']},
  ];

  return<div style={{display:'flex',minHeight:'100vh',background:T.bg,position:'relative',zIndex:2}}>
    <aside style={{
      width:260,background:'#0a0a0a',borderRight:`1px solid ${T.border}`,
      padding:'28px 18px 20px',display:'flex',flexDirection:'column',
      position:'fixed',top:0,left:0,bottom:0,zIndex:20,
      backdropFilter:'blur(20px)',overflowY:'auto',overflowX:'hidden',
    }}>
      {/* Animated top gradient line */}
      <div style={{position:'absolute',top:0,left:0,right:0,height:2,background:'linear-gradient(90deg,transparent,rgba(255,255,255,0.3),transparent)',animation:'topLine 4s ease-in-out infinite'}}/>
      {/* Grid texture */}
      <div style={{position:'absolute',inset:0,pointerEvents:'none',zIndex:0,backgroundImage:'linear-gradient(rgba(255,255,255,0.008) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,0.008) 1px,transparent 1px)',backgroundSize:'40px 40px',maskImage:'radial-gradient(ellipse 60% 80% at 50% 50%,rgba(0,0,0,0.4),transparent)',WebkitMaskImage:'radial-gradient(ellipse 60% 80% at 50% 50%,rgba(0,0,0,0.4),transparent)'}}/>

      {/* Logo */}
      <div style={{display:'flex',alignItems:'center',gap:12,padding:'0 8px',marginBottom:40,position:'relative',zIndex:2}}>
        <div style={{width:42,height:42,borderRadius:12,background:'linear-gradient(135deg,rgba(255,255,255,0.06),rgba(255,255,255,0.02))',border:`1px solid ${T.border}`,display:'flex',alignItems:'center',justifyContent:'center',animation:'logoGlow 4s ease-in-out infinite'}}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polygon points="23 7 16 12 23 17"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>
        </div>
        <div style={{display:'flex',flexDirection:'column',lineHeight:1.1}}>
          <div style={{fontSize:17,fontWeight:800,color:'#fff',letterSpacing:-0.5}}>Video<span style={{color:'rgba(255,255,255,0.25)',fontWeight:400}}>Resume</span><span style={{color:'rgba(255,255,255,0.5)'}}>AI</span></div>
          <div style={{fontSize:9,color:T.dim,fontFamily:"'JetBrains Mono',monospace",letterSpacing:1.5,textTransform:'uppercase',marginTop:3,fontWeight:500}}>v2.0 · BETA</div>
        </div>
      </div>

      <nav style={{flex:1,display:'flex',flexDirection:'column',gap:2,position:'relative',zIndex:2}}>
        {/* Dashboard section */}
        <div style={{fontSize:9,textTransform:'uppercase',letterSpacing:2,color:T.dim,padding:'0 14px 10px',fontWeight:700,display:'flex',alignItems:'center',gap:10}}>
          Dashboard<div style={{flex:1,height:1,background:`linear-gradient(90deg,${T.border},transparent)`}}/>
        </div>
        {navI.map((n,i)=><NavItem key={n.id} n={n} active={tab===n.id} onClick={()=>{setTab(n.id);setSelSub(null)}} badge={n.id==='submissions'&&subs.length>0?String(subs.length):(n.id==='analytics'?'new':null)} delay={0.1+i*0.05}/>)}

        {/* Actions section */}
        <div style={{fontSize:9,textTransform:'uppercase',letterSpacing:2,color:T.dim,padding:'22px 14px 10px',fontWeight:700,display:'flex',alignItems:'center',gap:10}}>
          Actions<div style={{flex:1,height:1,background:`linear-gradient(90deg,${T.border},transparent)`}}/>
        </div>
        <NavItem n={{l:'New Upload',paths:['M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4','M17 8l-5-5-5 5','M12 3v12']}} active={false} onClick={()=>go('upload')} delay={0.45}/>

        {/* Account section */}
        <div style={{fontSize:9,textTransform:'uppercase',letterSpacing:2,color:T.dim,padding:'22px 14px 10px',fontWeight:700,display:'flex',alignItems:'center',gap:10}}>
          Account<div style={{flex:1,height:1,background:`linear-gradient(90deg,${T.border},transparent)`}}/>
        </div>
        {accountNav.map((n,i)=><NavItem key={n.id} n={n} active={tab===n.id} onClick={()=>setTab(n.id)} delay={0.55+i*0.05}/>)}
      </nav>

      {/* User Card */}
      <div style={{marginTop:20,marginBottom:16,height:1,background:`linear-gradient(90deg,transparent,${T.border},transparent)`,position:'relative',zIndex:2}}/>
      <UserCard profile={profile} session={session} onLogout={onLogout}/>
    </aside>
    <main style={{flex:1,marginLeft:260,padding:'32px 44px',minHeight:'100vh'}}>
      {tab==='overview'&&<OvTab subs={subs} comp={comp} go={go} profile={profile} show={show} setTab={setTab} setSelSub={setSelSub}/>}
      {tab==='submissions'&&<SubTab subs={subs} show={show} ls={ls} session={session}/>}
      {tab==='resumes'&&<ResTab subs={comp}/>}
      {tab==='clips'&&<ClipTab subs={comp}/>}
      {tab==='analytics'&&<AnaTab subs={comp} selSub={selSub} setSelSub={setSelSub}/>}
      {tab==='prof'&&<ProfTab profile={profile} session={session} setProfile={setProfile} show={show}/>}
      {tab==='sett'&&<SetTab show={show}/>}
    </main>
  </div>;
};

/* User Card at bottom of sidebar */
const UserCard=({profile,session,onLogout})=>{
  const[hov,setHov]=useState(false);
  const[btnHov,setBtnHov]=useState(false);
  return <div style={{position:'relative',zIndex:2,padding:14,borderRadius:16,background:hov?'linear-gradient(135deg,rgba(255,255,255,0.05),rgba(255,255,255,0.02))':'linear-gradient(135deg,rgba(255,255,255,0.03),rgba(255,255,255,0.01))',border:`1px solid ${hov?'rgba(255,255,255,0.12)':T.border}`,transition:'all 0.4s',overflow:'hidden'}} onMouseEnter={()=>setHov(true)} onMouseLeave={()=>setHov(false)}>
    {/* Top shine line */}
    <div style={{position:'absolute',top:0,left:20,right:20,height:1,background:'linear-gradient(90deg,transparent,rgba(255,255,255,0.15),transparent)',opacity:hov?1:0,transition:'opacity 0.4s'}}/>
    <div style={{display:'flex',alignItems:'center',gap:12,marginBottom:14}}>
      <div style={{position:'relative',flexShrink:0}}>
        {profile?.avatar_url ? (
          <div style={{width:42,height:42,borderRadius:12,overflow:'hidden',border:`1px solid ${T.border}`,boxShadow:'0 4px 16px rgba(0,0,0,0.3)'}}>
            <img src={profile.avatar_url} alt="avatar" style={{width:'100%',height:'100%',objectFit:'cover',display:'block'}}/>
          </div>
        ) : (
          <div style={{width:42,height:42,borderRadius:12,background:'linear-gradient(135deg,#fff,#c4c4c4)',display:'flex',alignItems:'center',justifyContent:'center',color:'#060606',fontWeight:800,fontSize:17,boxShadow:'0 4px 16px rgba(255,255,255,0.08),inset 0 -2px 4px rgba(0,0,0,0.1)'}}>{profile?.full_name?.[0]?.toUpperCase()||'U'}</div>
        )}
        {/* Online pulse dot */}
        <div style={{position:'absolute',bottom:-2,right:-2,width:12,height:12,borderRadius:'50%',background:T.success,border:'2px solid #0a0a0a',boxShadow:'0 0 8px rgba(74,222,128,0.5)',animation:'pulseDot 2s ease-in-out infinite'}}/>
      </div>
      <div style={{flex:1,minWidth:0}}>
        <div style={{fontSize:13.5,fontWeight:700,color:'#fff',marginBottom:2,whiteSpace:'nowrap',overflow:'hidden',textOverflow:'ellipsis'}}>{profile?.full_name||'User'}</div>
        <div style={{fontSize:10.5,color:T.dim,whiteSpace:'nowrap',overflow:'hidden',textOverflow:'ellipsis',fontFamily:"'JetBrains Mono',monospace"}}>{session?.user?.email}</div>
      </div>
    </div>
    <button onClick={onLogout} onMouseEnter={()=>setBtnHov(true)} onMouseLeave={()=>setBtnHov(false)} style={{
      width:'100%',padding:10,borderRadius:10,
      background:btnHov?'rgba(248,113,113,0.05)':'rgba(255,255,255,0.02)',
      border:`1px solid ${btnHov?'rgba(248,113,113,0.2)':T.border}`,
      color:btnHov?T.danger:T.muted,fontSize:12,fontWeight:600,cursor:'pointer',
      fontFamily:T.font,transition:'all 0.3s',
      display:'flex',alignItems:'center',justifyContent:'center',gap:8,
    }}>
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{transform:btnHov?'translateX(3px)':'none',transition:'transform 0.3s'}}><path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>
      Sign Out
    </button>
  </div>;
};

/* Overview — 3D Stat Cards with tilt + ambient particles */
const TiltCard3D = ({icon, label, value, delay, sub, glowColor='rgba(255,255,255,0.06)'}) => {
  const ref = useRef();
  const [style, setStyle] = useState({});

  const handleMove = useCallback((e) => {
    const el = ref.current; if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const cx = rect.width/2, cy = rect.height/2;
    const rx = ((y-cy)/cy)*-8, ry = ((x-cx)/cx)*8;
    setStyle({
      transform: `translateY(-8px) rotateX(${rx}deg) rotateY(${ry}deg) scale(1.02)`,
      borderColor: 'rgba(255,255,255,0.12)',
      boxShadow: `0 20px 60px rgba(0,0,0,0.4), 0 0 40px rgba(255,255,255,0.02), inset 0 1px 0 rgba(255,255,255,0.06)`,
    });
  }, []);

  const handleLeave = useCallback(() => {
    setStyle({ transform: 'translateY(0) rotateX(0) rotateY(0) scale(1)', borderColor: T.border, boxShadow: 'none' });
  }, []);

  // Counter animation
  const valRef = useRef(); const [displayVal, setDisplayVal] = useState('0');
  useEffect(() => {
    const raw = String(value); const isP = raw.includes('%'); const num = parseInt(raw);
    if (isNaN(num)) { setDisplayVal(raw); return; }
    const dur = 1500; const start = performance.now();
    const tick = (now) => {
      const p = Math.min((now-start)/dur, 1); const eased = 1 - Math.pow(1-p, 4);
      setDisplayVal(Math.round(num * eased) + (isP ? '%' : ''));
      if (p < 1) requestAnimationFrame(tick);
    };
    const t = setTimeout(() => requestAnimationFrame(tick), delay * 1000 + 400);
    return () => clearTimeout(t);
  }, [value, delay]);

  // Mini bars data
  const bars = [8,14,10,18,12,22];

  return (
    <div ref={ref} onMouseMove={handleMove} onMouseLeave={handleLeave} style={{
      position:'relative', borderRadius:20, padding:'28px 24px',
      background:T.card, border:`1px solid ${T.border}`,
      backdropFilter:'blur(20px)', cursor:'default',
      transformStyle:'preserve-3d',
      transition:'all 0.5s cubic-bezier(0.23,1,0.32,1)',
      overflow:'hidden',
      animation:`cardEnter3D 0.8s cubic-bezier(0.34,1.56,0.64,1) ${delay}s both`,
      ...style,
    }}>
      {/* Holographic shine */}
      <div style={{position:'absolute',inset:0,borderRadius:20,background:'linear-gradient(135deg,transparent 0%,rgba(255,255,255,0.03) 40%,rgba(255,255,255,0.06) 50%,rgba(255,255,255,0.03) 60%,transparent 100%)',opacity:style.transform&&style.transform.includes('rotateX')?1:0,transition:'opacity 0.5s',pointerEvents:'none'}}/>
      {/* Top glow line */}
      <div style={{position:'absolute',top:0,left:24,right:24,height:1,background:'linear-gradient(90deg,transparent,rgba(255,255,255,0.15),transparent)',opacity:style.transform&&style.transform.includes('rotateX')?1:0,transition:'opacity 0.5s'}}/>
      {/* Inner glow orb */}
      <div style={{position:'absolute',width:120,height:120,borderRadius:'50%',top:-40,right:-30,filter:'blur(50px)',background:glowColor,opacity:style.transform&&style.transform.includes('rotateX')?1:0,transition:'opacity 0.6s',pointerEvents:'none'}}/>

      <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start',marginBottom:20,position:'relative',zIndex:1}}>
        <div style={{width:46,height:46,borderRadius:14,background:'rgba(255,255,255,0.04)',border:`1px solid ${T.border}`,display:'flex',alignItems:'center',justifyContent:'center',fontSize:19,transition:'all 0.5s cubic-bezier(0.34,1.56,0.64,1)',transform:style.transform&&style.transform.includes('rotateX')?'scale(1.15) rotate(-8deg)':'none'}}>{icon}</div>
        {sub&&<span style={{fontSize:10,color:T.dim,padding:'3px 10px',borderRadius:8,background:'rgba(255,255,255,0.02)',border:`1px solid ${T.border}`,letterSpacing:0.5,textTransform:'uppercase',fontWeight:500}}>{sub}</span>}
      </div>
      <div style={{fontSize:38,fontWeight:800,color:'#fff',letterSpacing:-2,marginBottom:4,position:'relative',zIndex:1,transition:'transform 0.5s',transform:style.transform&&style.transform.includes('rotateX')?'translateX(4px)':'none'}}>{displayVal}</div>
      <div style={{fontSize:13,color:T.muted,fontWeight:500,position:'relative',zIndex:1}}>{label}</div>

      {/* Mini bar chart */}
      <div style={{display:'flex',gap:3,alignItems:'flex-end',height:24,position:'absolute',bottom:20,right:20,opacity:style.transform&&style.transform.includes('rotateX')?0.3:0.12,transition:'opacity 0.5s'}}>
        {bars.map((h,i)=><div key={i} style={{width:4,height:h,borderRadius:2,background:'rgba(255,255,255,0.5)',animation:`barGrow 0.8s ease ${delay+0.1*i}s both`}}/>)}
      </div>
    </div>
  );
};

/* Ambient particles for overview */
const AmbientBg = () => {
  const canvasRef = useRef();
  useEffect(() => {
    const c = canvasRef.current; if (!c) return;
    const ctx = c.getContext('2d');
    let w = c.width = c.offsetWidth, h = c.height = c.offsetHeight;
    const particles = Array.from({length:35}, () => ({
      x: Math.random()*w, y: Math.random()*h,
      vx: (Math.random()-0.5)*0.3, vy: -0.2 - Math.random()*0.4,
      r: 1 + Math.random()*1.5, a: 0.05 + Math.random()*0.12,
    }));
    let raf;
    const draw = () => {
      ctx.clearRect(0,0,w,h);
      particles.forEach(p => {
        p.x += p.vx; p.y += p.vy;
        if (p.y < -10) { p.y = h+10; p.x = Math.random()*w; }
        if (p.x < -10) p.x = w+10; if (p.x > w+10) p.x = -10;
        ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
        ctx.fillStyle = `rgba(255,255,255,${p.a})`; ctx.fill();
      });
      raf = requestAnimationFrame(draw);
    };
    draw();
    const onResize = () => { w = c.width = c.offsetWidth; h = c.height = c.offsetHeight; };
    window.addEventListener('resize', onResize);
    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', onResize); };
  }, []);
  return <canvas ref={canvasRef} style={{position:'absolute',inset:0,width:'100%',height:'100%',pointerEvents:'none',zIndex:0}}/>;
};

/* Grid texture overlay */
const GridOverlay = () => (
  <div style={{position:'absolute',inset:0,pointerEvents:'none',zIndex:0,
    backgroundImage:'linear-gradient(rgba(255,255,255,0.012) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.012) 1px, transparent 1px)',
    backgroundSize:'60px 60px',
    maskImage:'radial-gradient(ellipse 80% 60% at 50% 40%, rgba(0,0,0,0.4), transparent)',
    WebkitMaskImage:'radial-gradient(ellipse 80% 60% at 50% 40%, rgba(0,0,0,0.4), transparent)',
  }}/>
);

/* Floating orbs */
const AmbientOrbs = () => (
  <>
    <div style={{position:'absolute',width:500,height:500,top:'-10%',right:'-5%',borderRadius:'50%',filter:'blur(100px)',background:'radial-gradient(circle,rgba(255,255,255,0.025),transparent 70%)',animation:'orbFloat 20s ease-in-out infinite, orbFadeIn 2s ease forwards',opacity:0,pointerEvents:'none',zIndex:0}}/>
    <div style={{position:'absolute',width:400,height:400,bottom:'-10%',left:'10%',borderRadius:'50%',filter:'blur(100px)',background:'radial-gradient(circle,rgba(255,255,255,0.015),transparent 70%)',animation:'orbFloat 20s ease-in-out infinite -7s, orbFadeIn 2s ease forwards',opacity:0,pointerEvents:'none',zIndex:0}}/>
    <div style={{position:'absolute',width:300,height:300,top:'40%',left:'50%',borderRadius:'50%',filter:'blur(100px)',background:'radial-gradient(circle,rgba(255,255,255,0.02),transparent 70%)',animation:'orbFloat 20s ease-in-out infinite -14s, orbFadeIn 2s ease forwards',opacity:0,pointerEvents:'none',zIndex:0}}/>
  </>
);

const OvTab=({subs,comp,go,profile,show,setTab,setSelSub})=>(
  <div style={{position:'relative',minHeight:'80vh'}}>
    {/* Ambient effects */}
    <AmbientBg/>
    <AmbientOrbs/>
    <GridOverlay/>

    {/* Content */}
    <div style={{position:'relative',zIndex:1}}>
      {/* Header */}
      <div style={{marginBottom:40,animation:'fadeUp .7s ease both'}}>
        <h1 style={{fontSize:32,fontWeight:800,letterSpacing:-1,marginBottom:6}}>Welcome back, <span style={{color:'rgba(255,255,255,0.35)',fontWeight:400}}>{profile?.full_name?.split(' ')[0]||'there'}</span></h1>
        <p style={{color:T.muted,fontSize:14}}>Your AI video resume dashboard</p>
      </div>

      {/* 3D Stat Cards */}
      <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:18,marginBottom:40,perspective:1200}}>
        <TiltCard3D icon="📹" label="Total Uploads" value={subs.length} delay={0.1} sub="all time" glowColor="rgba(255,255,255,0.06)"/>
        <TiltCard3D icon="📄" label="Resumes" value={comp.length} delay={0.2} sub="completed" glowColor="rgba(74,222,128,0.06)"/>
        <TiltCard3D icon="🎬" label="Clips" value={comp.filter(s=>s.highlight_clip_url).length} delay={0.3} sub="ready" glowColor="rgba(167,139,250,0.06)"/>
        <TiltCard3D icon="📊" label="Avg Score" value={comp.length?Math.round(comp.reduce((a,s)=>a+(s.confidence_score||0),0)/comp.length)+'%':'—'} delay={0.4} sub="confidence" glowColor="rgba(250,204,21,0.06)"/>
      </div>

      {/* CTA */}
      <div style={{marginBottom:36,animation:'cardReveal 0.7s ease 0.5s both'}}>
        <Btn onClick={()=>{if(subs.length>=10){show("Max 10","error");return}go('upload')}} full={false} style={{padding:'13px 30px',borderRadius:14,boxShadow:'0 2px 16px rgba(255,255,255,0.06)'}}>+ New Submission</Btn>
      </div>

      {/* Recent Activity header with line */}
      <div style={{display:'flex',alignItems:'center',gap:10,marginBottom:16,animation:'cardReveal 0.7s ease 0.6s both'}}>
        <h3 style={{fontSize:17,fontWeight:700,whiteSpace:'nowrap'}}>Recent Activity</h3>
        <div style={{flex:1,height:1,background:`linear-gradient(90deg,${T.border},transparent)`}}/>
      </div>

      {/* Activity items */}
      {subs.length===0
        ?<Card animate delay={0.7} style={{textAlign:'center',padding:48}}><p style={{color:T.dim}}>No submissions yet.</p></Card>
        :<div style={{display:'flex',flexDirection:'column',gap:8}}>
          {subs.slice(0,5).map((s,i)=>(
            <ActivityItem key={s.id} s={s} i={i} setTab={setTab} setSelSub={setSelSub}/>
          ))}
        </div>
      }
    </div>
  </div>
);

/* Activity item with hover effects */
const ActivityItem = ({s, i, setTab, setSelSub}) => {
  const [hov, setHov] = useState(false);
  return (
    <div onMouseEnter={()=>setHov(true)} onMouseLeave={()=>setHov(false)} style={{
      display:'flex', justifyContent:'space-between', alignItems:'center',
      padding:'16px 22px', borderRadius:16,
      background:T.card, border:`1px solid ${hov?'rgba(255,255,255,0.1)':T.border}`,
      backdropFilter:'blur(16px)',
      transition:'all 0.4s cubic-bezier(0.23,1,0.32,1)',
      animation:`itemSlide 0.6s ease ${0.7+i*0.1}s both`,
      cursor:'default', position:'relative', overflow:'hidden',
      transform: hov ? 'translateX(6px) translateY(-2px)' : 'none',
      boxShadow: hov ? '0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04)' : 'none',
    }}>
      {/* Left scan line on hover */}
      <div style={{position:'absolute',left:0,top:0,bottom:0,width:3,background:'linear-gradient(180deg,transparent,rgba(255,255,255,0.2),transparent)',opacity:hov?1:0,transition:'opacity 0.4s'}}/>

      <div style={{display:'flex',alignItems:'center',gap:14}}>
        <div style={{
          width:42,height:42,borderRadius:12,
          display:'flex',alignItems:'center',justifyContent:'center',
          fontSize:16,fontWeight:700,
          background:s.status==="completed"?'rgba(74,222,128,0.06)':s.status==="failed"?'rgba(248,113,113,0.06)':'rgba(255,255,255,0.04)',
          color:s.status==="completed"?T.success:s.status==="failed"?T.danger:'#fff',
          border:`1px solid ${s.status==="completed"?'rgba(74,222,128,0.1)':s.status==="failed"?'rgba(248,113,113,0.08)':T.border}`,
          transition:'all 0.4s',
          transform: hov ? 'scale(1.1) rotate(-5deg)' : 'none',
        }}>{s.status==="completed"?"✓":s.status==="failed"?"✗":"⏳"}</div>
        <div>
          <div style={{fontWeight:600,fontSize:14,marginBottom:2}}>{s.video_filename||"Video"}</div>
          <div style={{fontSize:11.5,color:T.dim}}>
            {new Date(s.created_at).toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric'})}
            {s.upload_method&&<span style={{marginLeft:8,padding:'2px 8px',borderRadius:5,fontSize:10,background:'rgba(255,255,255,0.02)',border:`1px solid ${T.border}`,color:T.muted}}>{s.upload_method==='record'?'🎥 Rec':'📁 Up'}</span>}
          </div>
        </div>
      </div>

      <div style={{display:'flex',gap:6,alignItems:'center'}}>
        {s.resume_pdf_url&&<a href={s.resume_pdf_url} target="_blank" rel="noopener" style={{padding:'6px 14px',borderRadius:8,fontSize:11.5,fontWeight:600,background:hov?'#fff':'rgba(255,255,255,0.04)',color:hov?'#0a0a0a':'#fff',border:`1px solid rgba(255,255,255,0.08)`,transition:'all 0.3s',textDecoration:'none',boxShadow:hov?'0 0 20px rgba(255,255,255,0.1)':'none'}}>Resume</a>}
        {s.highlight_clip_url&&<a href={s.highlight_clip_url} target="_blank" rel="noopener" style={{padding:'6px 14px',borderRadius:8,fontSize:11.5,fontWeight:600,background:'rgba(167,139,250,0.04)',color:T.purple,border:'1px solid rgba(167,139,250,0.08)',textDecoration:'none',transition:'all 0.3s'}}>Clip</a>}
        {s.status==="completed"&&<span onClick={()=>{setSelSub(s);setTab('analytics')}} style={{padding:'6px 14px',borderRadius:8,fontSize:11.5,fontWeight:600,background:'rgba(250,204,21,0.04)',color:T.warning,border:'1px solid rgba(250,204,21,0.08)',cursor:'pointer',transition:'all 0.3s'}}>Analytics</span>}
        <span style={{padding:'5px 12px',borderRadius:8,fontSize:10.5,fontWeight:600,background:s.status==="completed"?'rgba(74,222,128,0.05)':s.status==="failed"?'rgba(248,113,113,0.05)':'rgba(250,204,21,0.05)',color:s.status==="completed"?T.success:s.status==="failed"?T.danger:T.warning,textTransform:'capitalize',letterSpacing:0.3}}>{s.status}</span>
      </div>
    </div>
  );
};

/* Submissions */
const SubTab=({subs,show,ls,session})=>{const del=async id=>{if(!confirm('Delete?'))return;try{await db.deleteSubmission(id);show("Deleted","success");await ls(session.user.id)}catch(e){show(e.message,"error")}};return<div><div style={{marginBottom:28}}><h1 style={{fontSize:26,fontWeight:800}}>Submissions</h1><p style={{color:T.muted,fontSize:14,marginTop:4}}>Your full-length recorded videos</p></div>{subs.length===0?<Card animate style={{textAlign:'center',padding:56}}><p style={{color:T.dim}}>None yet.</p></Card>:<div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(380px,1fr))',gap:18}}>{subs.map((s,i)=><Card key={s.id} animate delay={i*0.08} style={{padding:0,overflow:'hidden'}}><div style={{background:'#000',position:'relative',overflow:'hidden'}}>{s.video_url?<video src={s.video_url} controls controlsList="nodownload" style={{width:'100%',display:'block',maxHeight:260}} preload="metadata"/>:<div style={{height:200,background:'linear-gradient(135deg,#111,#0b0b0b)',display:'flex',alignItems:'center',justifyContent:'center'}}><svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="1.5"><polygon points="23 7 16 12 23 17"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg></div>}<div style={{position:'absolute',top:12,right:12}}><span style={{padding:'4px 12px',borderRadius:8,fontSize:10.5,fontWeight:700,background:s.status==="completed"?'rgba(74,222,128,0.15)':s.status==="failed"?'rgba(248,113,113,0.15)':'rgba(250,204,21,0.15)',color:s.status==="completed"?T.success:s.status==="failed"?T.danger:T.warning,textTransform:'uppercase',letterSpacing:.5,backdropFilter:'blur(6px)'}}>{s.status}</span></div>{s.confidence_score>0&&<div style={{position:'absolute',top:12,left:12,padding:'4px 10px',borderRadius:8,background:'rgba(0,0,0,0.7)',backdropFilter:'blur(6px)',fontSize:12,fontWeight:700,color:'#fff'}}>{Math.round(s.confidence_score)}%</div>}<div style={{position:'absolute',bottom:12,left:12,padding:'3px 10px',borderRadius:6,background:'rgba(0,0,0,0.7)',backdropFilter:'blur(6px)',fontSize:10,fontWeight:600,color:'rgba(255,255,255,0.6)',letterSpacing:0.5}}>FULL VIDEO</div></div><div style={{padding:'18px 22px'}}><div style={{fontWeight:700,fontSize:15,marginBottom:4}}>{s.video_filename||"Video"}</div><div style={{fontSize:12,color:T.dim,marginBottom:14,display:'flex',alignItems:'center',gap:8}}>{new Date(s.created_at).toLocaleDateString('en-US',{month:'long',day:'numeric',year:'numeric'})}{s.upload_method&&<span style={{padding:'2px 8px',borderRadius:5,fontSize:10,background:'rgba(255,255,255,0.03)',border:`1px solid ${T.border}`,color:T.muted}}>{s.upload_method==='record'?'Recorded':'Uploaded'}</span>}</div><div style={{display:'flex',gap:8,flexWrap:'wrap'}}>{s.resume_pdf_url&&<a href={s.resume_pdf_url} target="_blank" rel="noopener" style={{padding:'8px 16px',borderRadius:8,fontSize:12,fontWeight:600,background:T.gradient,color:'#0b0b0b',textDecoration:'none'}}>Resume</a>}{s.video_url&&<a href={s.video_url} target="_blank" rel="noopener" style={{padding:'8px 16px',borderRadius:8,fontSize:12,fontWeight:600,background:'rgba(255,255,255,0.04)',color:'#fff',border:`1px solid ${T.border}`,textDecoration:'none',transition:'all 0.3s'}} onMouseEnter={e=>e.target.style.background='rgba(255,255,255,0.08)'} onMouseLeave={e=>e.target.style.background='rgba(255,255,255,0.04)'}>Open Video</a>}<button onClick={()=>del(s.id)} style={{padding:'8px 14px',borderRadius:8,fontSize:12,fontWeight:600,background:'rgba(248,113,113,0.04)',color:T.danger,border:'1px solid rgba(248,113,113,0.06)',cursor:'pointer',fontFamily:T.font,marginLeft:'auto'}}>Delete</button></div></div></Card>)}</div>}</div>};

/* Resumes */
const ResTab=({subs})=>{const rs=subs.filter(s=>s.resume_pdf_url);return<div><div style={{marginBottom:28}}><h1 style={{fontSize:26,fontWeight:800}}>Resumes</h1><p style={{color:T.muted,fontSize:14,marginTop:4}}>ATS-optimized PDF resumes</p></div>{rs.length===0?<Card animate style={{textAlign:'center',padding:56}}><p style={{color:T.dim}}>No resumes yet.</p></Card>:<div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(340px,1fr))',gap:24}}>{rs.map((s,i)=><ResumeCard3D key={s.id} s={s} i={i}/>)}</div>}</div>};

/* Share Row — WhatsApp, LinkedIn, Copy, Email */
const ShareRow=({url,title})=>{
  const[copied,setCopied]=useState(false);
  const msg=encodeURIComponent(`Check out my AI-generated resume: ${title}`);
  const encodedUrl=encodeURIComponent(url);

  const copyLink=()=>{
    navigator.clipboard.writeText(url).then(()=>{setCopied(true);setTimeout(()=>setCopied(false),2000)}).catch(()=>{});
  };

  const shareItems=[
    {label:'WhatsApp',href:`https://wa.me/?text=${msg}%20${encodedUrl}`,color:'#25D366',
      icon:<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 01-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 01-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 012.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0012.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 005.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893a11.821 11.821 0 00-3.48-8.413z"/></svg>},
    {label:'LinkedIn',href:`https://www.linkedin.com/sharing/share-offsite/?url=${encodedUrl}`,color:'#0A66C2',
      icon:<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>},
    {label:copied?'Copied!':'Copy Link',onClick:copyLink,color:copied?'#4ade80':'#fff',
      icon:<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">{copied?<><polyline points="20 6 9 17 4 12"/></>:<><path d="M10 13a5 5 0 007.54.54l3-3a5 5 0 00-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 00-7.54-.54l-3 3a5 5 0 007.07 7.07l1.71-1.71"/></>}</svg>},
    {label:'Email',href:`mailto:?subject=${msg}&body=${msg}%20${encodedUrl}`,color:'#fff',
      icon:<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><polyline points="22 6 12 13 2 6"/></svg>},
  ];

  return <div style={{display:'flex',gap:6,transform:'translateZ(6px)'}}>
    {shareItems.map(item=>{
      const[h,setH]=useState(false);
      const Tag=item.href?'a':'button';
      return <Tag key={item.label} href={item.href||undefined} target={item.href?'_blank':undefined} rel={item.href?'noopener':undefined} onClick={item.onClick||undefined}
        onMouseEnter={()=>setH(true)} onMouseLeave={()=>setH(false)}
        style={{
          flex:1,padding:'8px 0',borderRadius:10,fontSize:10.5,fontWeight:600,
          background:h?`${item.color}15`:'rgba(255,255,255,0.02)',
          color:h?item.color:'rgba(255,255,255,0.3)',
          border:`1px solid ${h?`${item.color}30`:'rgba(255,255,255,0.04)'}`,
          transition:'all 0.3s',display:'flex',alignItems:'center',justifyContent:'center',gap:5,
          textDecoration:'none',cursor:'pointer',fontFamily:T.font,
          transform:h?'translateY(-1px)':'none',
        }}>
        {item.icon}
        {item.label}
      </Tag>;
    })}
  </div>;
};

const ResumeCard3D=({s,i})=>{
  const[hov,setHov]=useState(false);
  const[tilt,setTilt]=useState({x:0,y:0});
  const[barW,setBarW]=useState(0);
  const ref=useRef(null);

  const score=Math.round(s.confidence_score||0);
  const scoreColor=score>=80?'#4ade80':score>=65?'#a78bfa':score>=50?'#facc15':'#f87171';
  const scoreDark=score>=80?'#16a34a':score>=65?'#7c3aed':score>=50?'#ca8a04':'#dc2626';
  const grade=score>=90?'A+':score>=80?'A':score>=70?'B+':score>=60?'B':score>=50?'C':'D';
  const circ=2*Math.PI*18;
  const dashOff=circ-(circ*score/100);

  useEffect(()=>{const t=setTimeout(()=>setBarW(score),400+i*100);return()=>clearTimeout(t)},[score,i]);

  const onMove=(e)=>{
    if(!ref.current)return;
    const r=ref.current.getBoundingClientRect();
    const x=((e.clientX-r.left)/r.width-0.5)*2;
    const y=((e.clientY-r.top)/r.height-0.5)*2;
    setTilt({x:x*8,y:-y*8});
  };

  const skills=(s.extracted_skills||[]).map(sk=>sk.skill_name||sk);

  return <div ref={ref} onMouseEnter={()=>setHov(true)} onMouseLeave={()=>{setHov(false);setTilt({x:0,y:0})}} onMouseMove={onMove}
    style={{perspective:1200,animation:`cardEnter3D 0.7s cubic-bezier(0.34,1.56,0.64,1) ${i*0.12}s both`}}>
    <div style={{
      position:'relative',borderRadius:22,padding:28,overflow:'hidden',
      background:'linear-gradient(165deg, rgba(22,22,25,0.95), rgba(10,10,12,0.98))',
      border:`1px solid ${hov?'rgba(255,255,255,0.14)':'rgba(255,255,255,0.06)'}`,
      transformStyle:'preserve-3d',
      transform:`rotateY(${tilt.x}deg) rotateX(${tilt.y}deg) ${hov?'translateY(-8px)':''}`,
      transition:tilt.x===0?'all 0.5s cubic-bezier(0.23,1,0.32,1)':'transform 0.12s ease-out',
      boxShadow:hov?'0 30px 80px rgba(0,0,0,0.6), 0 0 50px rgba(255,255,255,0.03)':'0 4px 12px rgba(0,0,0,0.3)',
    }}>

      {/* Accent line sweep */}
      <div style={{position:'absolute',top:0,left:0,right:0,height:2,background:'linear-gradient(90deg,transparent,rgba(255,255,255,0.5),transparent)',transform:hov?'translateX(0)':'translateX(-100%)',transition:'transform 0.8s cubic-bezier(0.23,1,0.32,1)'}}/>

      {/* Holographic shine */}
      <div style={{position:'absolute',inset:0,pointerEvents:'none',background:`radial-gradient(circle at ${50+tilt.x*5}% ${50+tilt.y*5}%, rgba(255,255,255,0.07) 0%, transparent 50%)`,opacity:hov?1:0,transition:'opacity 0.4s'}}/>

      {/* Circuit grid */}
      <div style={{position:'absolute',inset:0,pointerEvents:'none',opacity:hov?0.3:0,transition:'opacity 0.6s',backgroundImage:'linear-gradient(rgba(255,255,255,0.02) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,0.02) 1px,transparent 1px)',backgroundSize:'20px 20px',maskImage:'radial-gradient(ellipse 60% 50% at 80% 20%,rgba(0,0,0,0.5),transparent)',WebkitMaskImage:'radial-gradient(ellipse 60% 50% at 80% 20%,rgba(0,0,0,0.5),transparent)'}}/>

      {/* Header */}
      <div style={{display:'flex',alignItems:'flex-start',gap:16,marginBottom:22,position:'relative',transform:'translateZ(20px)'}}>
        {/* Document icon with corner fold */}
        <div style={{position:'relative',width:52,height:64,borderRadius:'4px 14px 14px 14px',background:'linear-gradient(145deg,rgba(255,255,255,0.07),rgba(255,255,255,0.02))',border:`1px solid ${hov?'rgba(255,255,255,0.18)':'rgba(255,255,255,0.08)'}`,display:'flex',alignItems:'center',justifyContent:'center',flexShrink:0,transition:'all 0.5s cubic-bezier(0.34,1.56,0.64,1)',transform:hov?'rotate(-5deg) scale(1.06)':'none',boxShadow:hov?'0 8px 24px rgba(255,255,255,0.06)':'none'}}>
          {/* Corner fold */}
          <div style={{position:'absolute',top:0,right:0,width:14,height:14,background:'linear-gradient(225deg,#0a0a0a 48%,rgba(255,255,255,0.1) 50%)',borderRadius:'0 14px 0 6px'}}/>
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" style={{opacity:hov?1:0.6,filter:hov?'drop-shadow(0 0 6px rgba(255,255,255,0.3))':'none',transition:'all 0.4s'}}>
            <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>
          </svg>
        </div>
        <div style={{flex:1,minWidth:0}}>
          <div style={{fontWeight:700,fontSize:15,color:'#fff',letterSpacing:-0.3,marginBottom:5,overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{s.video_filename||'Resume'}</div>
          <div style={{fontSize:11,color:T.dim,fontFamily:"'JetBrains Mono',monospace",letterSpacing:0.3,display:'flex',alignItems:'center',gap:6}}>
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" opacity="0.5"><rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
            {new Date(s.created_at).toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric'})}
          </div>
        </div>
      </div>

      {/* Score Row — separate section, no overlap */}
      {score>0&&<div style={{display:'flex',alignItems:'center',gap:14,marginBottom:20,padding:'12px 16px',borderRadius:14,background:hov?'rgba(255,255,255,0.03)':'rgba(255,255,255,0.02)',border:`1px solid ${hov?'rgba(255,255,255,0.08)':'rgba(255,255,255,0.04)'}`,transform:'translateZ(15px)',transition:'all 0.4s'}}>
        {/* Mini gauge ring */}
        <div style={{position:'relative',width:44,height:44,flexShrink:0}}>
          <svg width="44" height="44" viewBox="0 0 44 44" style={{transform:'rotate(-90deg)'}}>
            <circle cx="22" cy="22" r="18" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="3"/>
            <circle cx="22" cy="22" r="18" fill="none" stroke={scoreColor} strokeWidth="3" strokeLinecap="round" strokeDasharray={circ} strokeDashoffset={dashOff} style={{transition:'stroke-dashoffset 1.5s cubic-bezier(0.23,1,0.32,1)'}}/>
          </svg>
          <div style={{position:'absolute',inset:0,display:'flex',alignItems:'center',justifyContent:'center',fontSize:13,fontWeight:800,color:scoreColor,fontFamily:"'JetBrains Mono',monospace"}}>{score}</div>
        </div>
        {/* Bar */}
        <div style={{flex:1}}>
          <div style={{fontSize:10,color:'rgba(255,255,255,0.35)',textTransform:'uppercase',letterSpacing:1,fontWeight:600,marginBottom:4}}>Confidence Score</div>
          <div style={{height:4,borderRadius:3,background:'rgba(255,255,255,0.04)',overflow:'visible',position:'relative'}}>
            <div style={{height:'100%',borderRadius:3,width:`${barW}%`,background:`linear-gradient(90deg,${scoreDark},${scoreColor},${scoreDark},${scoreColor})`,backgroundSize:'200% 100%',animation:'barShimmer 2s ease-in-out infinite',transition:'width 1.2s cubic-bezier(0.23,1,0.32,1)',position:'relative'}}>
              <div style={{position:'absolute',right:-2,top:'50%',transform:'translateY(-50%)',width:10,height:10,borderRadius:'50%',background:scoreColor,filter:'blur(4px)',opacity:barW>0?1:0,animation:'barPulse 1.5s ease-in-out infinite'}}/>
            </div>
          </div>
        </div>
        {/* Grade */}
        <div style={{fontSize:20,fontWeight:800,color:scoreColor,fontFamily:"'JetBrains Mono',monospace",width:40,textAlign:'center'}}>{grade}</div>
      </div>}

      {/* Skills */}
      {skills.length>0&&<div style={{display:'flex',gap:6,flexWrap:'wrap',marginBottom:22,transform:'translateZ(12px)'}}>
        {skills.slice(0,5).map((sk,j)=><span key={j} style={{padding:'5px 12px',borderRadius:7,fontSize:10.5,fontWeight:500,background:hov?'rgba(255,255,255,0.04)':'rgba(255,255,255,0.02)',border:`1px solid ${hov?'rgba(255,255,255,0.1)':'rgba(255,255,255,0.05)'}`,color:hov?'rgba(255,255,255,0.8)':'rgba(255,255,255,0.45)',letterSpacing:0.2,transition:`all 0.35s ease ${j*0.06}s`,animation:hov?`skillGlow 0.5s ease ${j*0.06}s both`:'none'}}>{sk}</span>)}
        {skills.length>5&&<span style={{padding:'5px 10px',fontSize:10.5,color:T.dim,fontWeight:500}}>+{skills.length-5}</span>}
      </div>}

      {/* Buttons */}
      <div style={{display:'flex',gap:10,transform:'translateZ(8px)',marginBottom:12}}>
        <a href={s.resume_pdf_url} download style={{flex:1,padding:'12px 0',borderRadius:12,fontSize:12.5,fontWeight:700,background:hov?'#fff':'rgba(255,255,255,0.05)',color:hov?'#0a0a0a':'#fff',border:`1px solid ${hov?'#fff':'rgba(255,255,255,0.08)'}`,textAlign:'center',transition:'all 0.4s cubic-bezier(0.23,1,0.32,1)',display:'flex',alignItems:'center',justifyContent:'center',gap:7,boxShadow:hov?'0 8px 24px rgba(255,255,255,0.15)':'none',textDecoration:'none'}}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
          Download
        </a>
        <a href={s.resume_pdf_url} target="_blank" rel="noopener" style={{flex:1,padding:'12px 0',borderRadius:12,fontSize:12.5,fontWeight:600,background:'rgba(255,255,255,0.02)',color:hov?'#fff':T.muted,border:`1px solid ${hov?'rgba(255,255,255,0.12)':'rgba(255,255,255,0.06)'}`,textAlign:'center',transition:'all 0.4s',display:'flex',alignItems:'center',justifyContent:'center',gap:7,textDecoration:'none'}}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
          View
        </a>
      </div>

      {/* Share buttons */}
      <ShareRow url={s.resume_pdf_url} title={s.video_filename||'My AI Resume'}/>

      {/* Corner accent */}
      <div style={{position:'absolute',bottom:0,right:0,width:80,height:80,pointerEvents:'none',background:'radial-gradient(circle at bottom right, rgba(255,255,255,0.05), transparent 70%)',opacity:hov?1:0,transition:'opacity 0.5s'}}/>
    </div>
  </div>;
};

/* Clips */
const ClipTab=({subs})=>{const cs=subs.filter(s=>s.highlight_clip_url);return<div><div style={{marginBottom:28}}><h1 style={{fontSize:26,fontWeight:800}}>Highlight Clips</h1></div>{cs.length===0?<Card animate style={{textAlign:'center',padding:56}}><p style={{color:T.dim}}>No clips yet.</p></Card>:<div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(400px,1fr))',gap:18}}>{cs.map((s,i)=><Card key={s.id} animate delay={i*0.1} style={{padding:0,overflow:'hidden'}}><div style={{background:'#000',aspectRatio:'16/9'}}><video src={s.highlight_clip_url} controls style={{width:'100%',height:'100%',objectFit:'cover'}}/></div><div style={{padding:'18px 20px'}}><div style={{fontWeight:700,fontSize:15,marginBottom:6}}>{s.video_filename||'Highlight'}</div><div style={{display:'flex',gap:8,marginBottom:14,flexWrap:'wrap'}}>{s.highlight_start!=null&&s.highlight_end!=null&&<span style={{padding:'5px 12px',borderRadius:8,fontSize:11,fontWeight:600,background:'rgba(255,255,255,0.03)',color:'#fff',border:`1px solid ${T.border}`}}>⏱ {fmtTime(s.highlight_start)} — {fmtTime(s.highlight_end)}</span>}{s.confidence_score>0&&<span style={{padding:'5px 12px',borderRadius:8,fontSize:11,fontWeight:600,background:'rgba(74,222,128,0.04)',color:T.success,border:'1px solid rgba(74,222,128,0.06)'}}>Confidence: {Math.round(s.confidence_score)}%</span>}</div><a href={s.highlight_clip_url} download style={{padding:'8px 18px',borderRadius:9,fontSize:12,fontWeight:600,background:T.gradient,color:'#0b0b0b',display:'inline-block'}}>⬇ Download</a></div></Card>)}</div>}</div>};

/* ================================================================
   ANALYTICS — Interactive Dashboard
   ================================================================ */

/* Performance Bar */
const PerfBar=({label,value,color,rank,avg,delay})=>{
  const[w,setW]=useState(0);
  const[counter,setCounter]=useState(0);
  const mounted=useRef(false);
  useEffect(()=>{
    const t1=setTimeout(()=>setW(value),400+delay*120);
    const t2=setTimeout(()=>{
      mounted.current=true;
      const start=performance.now();
      const tick=(now)=>{const p=Math.min((now-start)/1200,1);setCounter(Math.round(value*(1-Math.pow(1-p,4))));if(p<1)requestAnimationFrame(tick)};
      requestAnimationFrame(tick);
    },400+delay*120);
    return()=>{clearTimeout(t1);clearTimeout(t2)};
  },[value,delay]);
  return <div style={{marginBottom:22,animation:`cardIn 0.5s ease ${0.2+delay*0.12}s both`}}>
    <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:8}}>
      <div style={{fontSize:11,color:T.textSoft,fontFamily:"'JetBrains Mono',monospace",textTransform:'uppercase',letterSpacing:1,fontWeight:500,display:'flex',alignItems:'center',gap:8}}>
        <span style={{width:20,height:20,borderRadius:6,display:'flex',alignItems:'center',justifyContent:'center',fontSize:9,fontWeight:800,color,border:`1px solid ${color}30`,background:`${color}08`}}>#{rank}</span>
        {label}
      </div>
      <span style={{fontSize:18,fontWeight:800,fontFamily:"'JetBrains Mono',monospace",color,letterSpacing:-0.5}}>{counter}%</span>
    </div>
    <div style={{height:6,borderRadius:6,background:'rgba(255,255,255,0.03)',position:'relative',overflow:'visible'}}>
      <div style={{height:'100%',borderRadius:6,width:`${w}%`,background:`linear-gradient(90deg,${color}40,${color})`,transition:'width 1.4s cubic-bezier(0.23,1,0.32,1)',position:'relative'}}>
        <div style={{position:'absolute',right:-3,top:'50%',transform:'translateY(-50%)',width:10,height:10,borderRadius:'50%',background:color,filter:'blur(4px)',opacity:w>0?0.8:0,transition:'opacity 0.5s 1s'}}/>
      </div>
      <div style={{position:'absolute',top:-2,bottom:-2,width:1,background:'rgba(255,255,255,0.12)',left:`${avg}%`,transition:'left 1s ease 0.5s'}}/>
    </div>
  </div>;
};

/* Gauge Ring */
const GaugeRing=({label,value,color,delay})=>{
  const circ=2*Math.PI*32;
  const[offset,setOffset]=useState(circ);
  const[counter,setCounter]=useState(0);
  useEffect(()=>{
    const t=setTimeout(()=>{
      setOffset(circ-(circ*value/100));
      const start=performance.now();
      const tick=(now)=>{const p=Math.min((now-start)/1200,1);setCounter(Math.round(value*(1-Math.pow(1-p,4))));if(p<1)requestAnimationFrame(tick)};
      requestAnimationFrame(tick);
    },500+delay*100);
    return()=>clearTimeout(t);
  },[value,delay]);
  return <div style={{textAlign:'center',width:80,animation:`cardIn 0.5s ease ${0.3+delay*0.1}s both`}}>
    <div style={{position:'relative',width:80,height:80}}>
      <svg width="80" height="80" viewBox="0 0 80 80" style={{transform:'rotate(-90deg)'}}>
        <circle cx="40" cy="40" r="32" fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="5"/>
        <circle cx="40" cy="40" r="32" fill="none" stroke={color} strokeWidth="5" strokeLinecap="round" strokeDasharray={circ} strokeDashoffset={offset} style={{transition:'stroke-dashoffset 1.5s cubic-bezier(0.4,0,0.2,1)'}}/>
      </svg>
      <div style={{position:'absolute',top:'50%',left:'50%',transform:'translate(-50%,-55%)',fontSize:16,fontWeight:800,fontFamily:"'JetBrains Mono',monospace",color:'#fff'}}>{counter}</div>
    </div>
    <div style={{marginTop:8,fontSize:10,color:T.dim,textTransform:'uppercase',letterSpacing:0.8,fontWeight:500}}>{label}</div>
  </div>;
};

/* Bar Chart Item */
const BarItem=({label,value,color,delay})=>{
  const[h,setH]=useState(0);
  useEffect(()=>{const t=setTimeout(()=>setH(value),600+delay*150);return()=>clearTimeout(t)},[value,delay]);
  return <div style={{flex:1,display:'flex',flexDirection:'column',alignItems:'center',gap:6}}>
    <div style={{width:'100%',height:140,background:'rgba(255,255,255,0.02)',borderRadius:8,position:'relative',display:'flex',alignItems:'flex-end',overflow:'hidden'}}>
      <div style={{width:'100%',borderRadius:'8px 8px 0 0',background:color,height:`${h}%`,transition:'height 1.2s cubic-bezier(0.34,1.56,0.64,1)',position:'relative'}}>
        <span style={{position:'absolute',top:-20,left:'50%',transform:'translateX(-50%)',fontSize:11,fontWeight:800,fontFamily:"'JetBrains Mono',monospace",color:'#fff'}}>{value}%</span>
      </div>
    </div>
    <span style={{fontSize:9,color:T.dim,textTransform:'uppercase',letterSpacing:0.5}}>{label}</span>
  </div>;
};

/* Sparkline */
const Sparkline=({data})=>{
  const w=500,h=80,pad=4;
  if(!data||!data.length) return null;
  const min=Math.min(...data),max=Math.max(...data),range=max-min||1;
  const stepX=(w-pad*2)/(data.length-1);
  let linePts='',areaPts=`${pad},${h-pad} `;
  data.forEach((v,i)=>{const x=pad+i*stepX,y=h-pad-((v-min)/range)*(h-pad*2);linePts+=`${x},${y} `;areaPts+=`${x},${y} `});
  areaPts+=`${pad+(data.length-1)*stepX},${h-pad}`;
  return <svg viewBox={`0 0 ${w} ${h}`} style={{width:'100%',height:60}} preserveAspectRatio="none">
    <polygon points={areaPts} fill="rgba(255,255,255,0.03)"/>
    <polyline points={linePts} fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round"/>
    {data.map((v,i)=>{const x=pad+i*stepX,y=h-pad-((v-min)/range)*(h-pad*2);return <circle key={i} cx={x} cy={y} r="3" fill="#fff" opacity="0" style={{transition:'opacity 0.3s'}}/>})}
  </svg>;
};

/* Skills Cloud */
const SkillTag=({skill,delay})=>{
  const[hov,setHov]=useState(false);
  return <span onMouseEnter={()=>setHov(true)} onMouseLeave={()=>setHov(false)} style={{
    padding:'6px 14px',borderRadius:8,fontSize:11,fontWeight:500,
    background:hov?'rgba(255,255,255,0.06)':'rgba(255,255,255,0.02)',
    border:`1px solid ${hov?'rgba(255,255,255,0.12)':T.border}`,
    color:hov?'#fff':T.muted,transition:'all 0.3s',cursor:'default',
    transform:hov?'translateY(-2px)':'none',
    animation:`cardIn 0.4s cubic-bezier(0.34,1.56,0.64,1) ${0.5+delay*0.06}s both`,
  }}>{skill}</span>;
};

/* Stat Mini Card */
const StatMini=({value,label})=>{
  const[hov,setHov]=useState(false);
  return <div onMouseEnter={()=>setHov(true)} onMouseLeave={()=>setHov(false)} style={{
    flex:1,padding:16,borderRadius:14,textAlign:'center',
    background:hov?'rgba(255,255,255,0.03)':'rgba(255,255,255,0.015)',
    border:`1px solid ${hov?'rgba(255,255,255,0.08)':T.border}`,
    transition:'all 0.3s',transform:hov?'translateY(-2px)':'none',
  }}>
    <div style={{fontSize:24,fontWeight:800,fontFamily:"'JetBrains Mono',monospace",color:'#fff',letterSpacing:-1}}>{value}</div>
    <div style={{fontSize:10,color:T.dim,marginTop:4,textTransform:'uppercase',letterSpacing:0.8}}>{label}</div>
  </div>;
};

/* Main Analytics Tab — DIFFERENTIATED CARDS */
const AnaTab=({subs,selSub,setSelSub})=>{
  const cur=selSub||subs[0];

  // Sort metrics by value for ranking
  const getMetrics=(s)=>{
    if(!s) return [];
    return [
      {label:'Confidence',value:s.confidence_score||0,color:'#ffffff'},
      {label:'Expression',value:s.expression_score||0,color:'#a78bfa'},
      {label:'Eye Contact',value:s.eye_contact_score||0,color:'#4ade80'},
      {label:'Energy',value:Math.min(100,s.energy_score||0),color:'#facc15'},
    ].sort((a,b)=>b.value-a.value).map((m,i)=>({...m,rank:i+1}));
  };

  const metrics=getMetrics(cur);
  const avg=cur?Math.round(metrics.reduce((a,m)=>a+m.value,0)/metrics.length):0;
  const skills=(cur?.extracted_skills||[]).map(sk=>sk.skill_name||sk);
  const timeline=Array.from({length:16},()=>30+Math.random()*50);

  // Computed insights
  const strongest=metrics[0];
  const weakest=metrics[metrics.length-1];
  const grade=avg>=80?'A':avg>=65?'B':avg>=50?'C':avg>=35?'D':'F';
  const gradeColor=avg>=80?'#4ade80':avg>=65?'#fff':avg>=50?'#facc15':'#f87171';
  const wordCount=cur?.transcript?cur.transcript.split(/\s+/).length:0;
  const clipDur=cur?.highlight_end&&cur?.highlight_start?Math.round(cur.highlight_end-cur.highlight_start):30;
  const fileSize=cur?.video_size_bytes?((cur.video_size_bytes)/(1024*1024)).toFixed(1):'-';

  // Tips based on weakest areas
  const getTips=(s)=>{
    if(!s) return [];
    const tips=[];
    if((s.confidence_score||0)<60) tips.push({tip:'Speak clearly and maintain a steady pace',area:'Confidence'});
    if((s.expression_score||0)<60) tips.push({tip:'Smile naturally and use hand gestures',area:'Expression'});
    if((s.eye_contact_score||0)<60) tips.push({tip:'Look directly at the camera lens',area:'Eye Contact'});
    if((s.energy_score||0)<60) tips.push({tip:'Project your voice and show enthusiasm',area:'Energy'});
    if(tips.length===0) tips.push({tip:'Great performance! Try varying your vocal tone for even more impact',area:'Overall'});
    return tips;
  };
  const tips=getTips(cur);

  const CTitle=({icon,label})=><div style={{fontSize:13,fontWeight:600,color:T.muted,marginBottom:18,textTransform:'uppercase',letterSpacing:1,display:'flex',alignItems:'center',gap:8}}><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" opacity="0.4"><path d={icon}/></svg>{label}</div>;

  return <div>
    <div style={{marginBottom:32,animation:'enterUp 0.5s ease both'}}><h1 style={{fontSize:30,fontWeight:800,letterSpacing:-1.5,marginBottom:6}}>Analytics</h1><p style={{color:T.muted,fontSize:14}}>Interactive performance breakdown per submission</p></div>

    {subs.length===0?<Card animate style={{textAlign:'center',padding:56}}><p style={{color:T.dim}}>No completed submissions to analyze.</p></Card>:<>

    {/* Submission Selector Tabs */}
    <div style={{display:'flex',gap:6,marginBottom:28,flexWrap:'wrap',animation:'enterUp 0.5s ease 0.1s both'}}>
      {subs.map((s,i)=><button key={s.id} onClick={()=>setSelSub(s)} style={{padding:'9px 18px',borderRadius:10,fontSize:12,fontWeight:600,fontFamily:T.font,cursor:'pointer',transition:'all 0.3s',border:`1px solid ${cur?.id===s.id?'rgba(255,255,255,0.12)':T.border}`,background:cur?.id===s.id?'rgba(255,255,255,0.06)':'rgba(255,255,255,0.01)',color:cur?.id===s.id?'#fff':T.muted}}>{s.video_filename||`Submission ${i+1}`}</button>)}
    </div>

    {cur&&<div style={{display:'grid',gridTemplateColumns:'1fr 1fr 1fr',gap:18}}>

      {/* 1. PERFORMANCE RANKING — sorted bars with rank badges */}
      <Card animate delay={0.15} style={{padding:28}}>
        <CTitle icon="M22 12h-4l-3 9L9 3l-3 9H2" label="Performance Ranking"/>
        {metrics.map((m,i)=><PerfBar key={m.label} label={m.label} value={m.value} color={m.color} rank={m.rank} avg={avg} delay={i}/>)}
      </Card>

      {/* 2. INTERVIEW READINESS — overall grade + readiness gauge */}
      <Card animate delay={0.25} style={{padding:28}}>
        <CTitle icon="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" label="Interview Readiness"/>
        <div style={{display:'flex',flexDirection:'column',alignItems:'center',gap:16}}>
          <div style={{position:'relative',width:120,height:120}}>
            <GaugeRing label="" value={avg} color={gradeColor} delay={0}/>
          </div>
          <div style={{textAlign:'center'}}>
            <div style={{fontSize:48,fontWeight:800,fontFamily:"'JetBrains Mono',monospace",color:gradeColor,lineHeight:1}}>{grade}</div>
            <div style={{fontSize:11,color:T.dim,marginTop:4,textTransform:'uppercase',letterSpacing:1}}>{avg>=80?'Interview Ready':avg>=65?'Almost There':avg>=50?'Needs Work':'Keep Practicing'}</div>
          </div>
          <div style={{width:'100%',padding:'12px 16px',borderRadius:12,background:'rgba(255,255,255,0.02)',border:`1px solid ${T.border}`,display:'flex',justifyContent:'space-between',fontSize:12}}>
            <span style={{color:T.dim}}>Strongest</span>
            <span style={{color:strongest?.color,fontWeight:700}}>{strongest?.label} ({strongest?.value}%)</span>
          </div>
        </div>
      </Card>

      {/* 3. IMPROVEMENT TIPS — AI-generated advice */}
      <Card animate delay={0.35} style={{padding:28}}>
        <CTitle icon="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" label="Improvement Tips"/>
        <div style={{display:'flex',flexDirection:'column',gap:12}}>
          {tips.map((t,i)=><div key={i} style={{padding:'14px 16px',borderRadius:12,background:'rgba(255,255,255,0.02)',border:`1px solid ${T.border}`,animation:`cardIn 0.4s ease ${0.4+i*0.1}s both`}}>
            <div style={{fontSize:10,fontWeight:700,color:T.warning,textTransform:'uppercase',letterSpacing:0.8,marginBottom:6}}>{t.area}</div>
            <div style={{fontSize:13,color:T.textSoft,lineHeight:1.5}}>{t.tip}</div>
          </div>)}
          {weakest&&<div style={{padding:'10px 16px',borderRadius:10,background:'rgba(248,113,113,0.04)',border:'1px solid rgba(248,113,113,0.08)',fontSize:12,color:'rgba(248,113,113,0.8)'}}>Focus area: <strong>{weakest.label}</strong> ({weakest.value}%)</div>}
        </div>
      </Card>

      {/* 4. SUBMISSION SUMMARY — unique stats, not same 4 scores */}
      <Card animate delay={0.45} style={{padding:28,gridColumn:'span 2'}}>
        <CTitle icon="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" label="Submission Summary"/>
        <div style={{display:'flex',gap:14}}>
          <StatMini value={`${avg}%`} label="Overall Score"/>
          <StatMini value={grade} label="Grade"/>
          <StatMini value={`${wordCount}`} label="Words Spoken"/>
          <StatMini value={`${clipDur}s`} label="Clip Duration"/>
          <StatMini value={`${fileSize}MB`} label="Video Size"/>
        </div>
      </Card>

      {/* 5. SKILLS CLOUD */}
      <Card animate delay={0.55} style={{padding:28}}>
        <CTitle icon="M20.59 13.41l-7.17 7.17a2 2 0 01-2.83 0L2 12V2h10l8.59 8.59a2 2 0 010 2.82zM7 7h.01" label="Extracted Skills"/>
        <div style={{display:'flex',flexWrap:'wrap',gap:6}}>
          {skills.length>0?skills.map((sk,i)=><SkillTag key={sk} skill={sk} delay={i}/>):<span style={{color:T.dim,fontSize:13}}>No skills extracted</span>}
        </div>
      </Card>

      {/* 6. EXPRESSION TIMELINE — spans 2 cols */}
      <Card animate delay={0.65} style={{padding:28,gridColumn:'span 2'}}>
        <CTitle icon="M22 12h-4l-3 9L9 3l-3 9H2" label="Expression Timeline"/>
        <Sparkline data={timeline}/>
      </Card>

      {/* 7. TRANSCRIPT */}
      <Card animate delay={0.75} style={{padding:28}}>
        <CTitle icon="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" label="Transcript"/>
        {cur.transcript?<div style={{fontSize:13,color:T.textSoft,lineHeight:1.9,maxHeight:180,overflowY:'auto',padding:'14px 18px',borderRadius:14,background:'rgba(0,0,0,0.3)',border:`1px solid ${T.border}`}}>{cur.transcript}</div>
        :<span style={{color:T.dim,fontSize:13}}>No transcript available</span>}
      </Card>

    </div>}
    </>}
  </div>;
};

/* Profile & Settings */
/* Profile Tab with Avatar Upload */
const ProfTab=({profile,session,setProfile,show})=>{
  const[uploading,setUploading]=useState(false);
  const[avHov,setAvHov]=useState(false);
  const fileRef=useRef();

  const handleFile=async(e)=>{
    const file=e.target.files?.[0];
    if(!file)return;
    // Validate
    if(!file.type.startsWith('image/')){show("Please select an image","error");return}
    if(file.size>5*1024*1024){show("Max 5MB","error");return}
    setUploading(true);
    try{
      // Upload to Cloudinary as image
      const CLOUD_NAME=import.meta.env.VITE_CLOUDINARY_CLOUD_NAME;
      const UPLOAD_PRESET=import.meta.env.VITE_CLOUDINARY_UPLOAD_PRESET||'video_resume_upload';
      const fd=new FormData();
      fd.append('file',file);
      fd.append('upload_preset',UPLOAD_PRESET);
      const res=await fetch(`https://api.cloudinary.com/v1_1/${CLOUD_NAME}/image/upload`,{method:'POST',body:fd});
      if(!res.ok)throw new Error("Cloudinary upload failed");
      const data=await res.json();
      // Save URL to profile
      const updated=await db.updateProfile(session.user.id,{avatar_url:data.secure_url});
      setProfile(updated);
      show("Profile picture updated!","success");
    }catch(err){
      console.error(err);
      show(err.message||"Upload failed","error");
    }finally{
      setUploading(false);
      if(fileRef.current)fileRef.current.value='';
    }
  };

  const handleRemove=async()=>{
    if(!confirm("Remove profile picture?"))return;
    try{
      const updated=await db.updateProfile(session.user.id,{avatar_url:null});
      setProfile(updated);
      show("Profile picture removed","success");
    }catch(err){
      show(err.message||"Failed","error");
    }
  };

  return <div>
    <div style={{marginBottom:32,animation:'fadeUp 0.6s ease both'}}>
      <h1 style={{fontSize:30,fontWeight:800,letterSpacing:-1,marginBottom:6}}>Profile</h1>
      <p style={{color:T.muted,fontSize:14}}>Manage your personal information</p>
    </div>

    <Card animate style={{padding:36,maxWidth:620,position:'relative',overflow:'visible'}} glow>
      {/* Avatar Section */}
      <div style={{display:'flex',alignItems:'center',gap:24,marginBottom:32,paddingBottom:28,borderBottom:`1px solid ${T.border}`}}>
        <div
          onMouseEnter={()=>setAvHov(true)}
          onMouseLeave={()=>setAvHov(false)}
          onClick={()=>!uploading&&fileRef.current?.click()}
          style={{
            position:'relative',width:110,height:110,borderRadius:24,cursor:uploading?'wait':'pointer',
            transition:'all 0.4s cubic-bezier(0.34,1.56,0.64,1)',
            transform:avHov&&!uploading?'scale(1.03)':'scale(1)',
            animation:'cardReveal 0.6s ease both',
          }}
        >
          {/* Avatar display */}
          {profile?.avatar_url ? (
            <div style={{width:'100%',height:'100%',borderRadius:24,overflow:'hidden',border:`2px solid ${avHov?'rgba(255,255,255,0.2)':T.border}`,transition:'border-color 0.3s',boxShadow:avHov?'0 20px 60px rgba(0,0,0,0.5), 0 0 40px rgba(255,255,255,0.08)':'0 8px 24px rgba(0,0,0,0.3)'}}>
              <img src={profile.avatar_url} alt="avatar" style={{width:'100%',height:'100%',objectFit:'cover',display:'block'}}/>
            </div>
          ) : (
            <div style={{width:'100%',height:'100%',borderRadius:24,background:'linear-gradient(135deg,#fff,#c4c4c4)',display:'flex',alignItems:'center',justifyContent:'center',fontSize:42,color:'#060606',fontWeight:800,border:`2px solid ${avHov?'rgba(255,255,255,0.3)':T.border}`,transition:'all 0.3s',boxShadow:avHov?'0 20px 60px rgba(0,0,0,0.5), 0 0 40px rgba(255,255,255,0.1)':'0 8px 24px rgba(0,0,0,0.3)'}}>
              {profile?.full_name?.[0]?.toUpperCase()||'U'}
            </div>
          )}

          {/* Hover overlay */}
          <div style={{
            position:'absolute',inset:0,borderRadius:24,
            background:'rgba(0,0,0,0.6)',backdropFilter:'blur(4px)',
            display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',gap:6,
            opacity:avHov&&!uploading?1:0,transition:'opacity 0.3s',pointerEvents:'none',
          }}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z"/>
              <circle cx="12" cy="13" r="4"/>
            </svg>
            <span style={{fontSize:10,color:'#fff',fontWeight:600,letterSpacing:0.5,textTransform:'uppercase'}}>Change</span>
          </div>

          {/* Upload spinner */}
          {uploading && (
            <div style={{position:'absolute',inset:0,borderRadius:24,background:'rgba(0,0,0,0.8)',backdropFilter:'blur(6px)',display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',gap:8}}>
              <div style={{width:32,height:32,border:`3px solid ${T.border}`,borderTopColor:'#fff',borderRadius:'50%',animation:'spin .8s linear infinite'}}/>
              <span style={{fontSize:10,color:T.muted,fontWeight:600}}>Uploading...</span>
            </div>
          )}

          {/* Camera indicator badge (bottom-right) */}
          {!uploading && (
            <div style={{position:'absolute',bottom:-4,right:-4,width:32,height:32,borderRadius:10,background:'#fff',display:'flex',alignItems:'center',justifyContent:'center',border:'3px solid #0b0b0b',boxShadow:'0 4px 12px rgba(0,0,0,0.4)',transition:'transform 0.3s',transform:avHov?'scale(1.1)':'scale(1)'}}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#0b0b0b" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z"/>
                <circle cx="12" cy="13" r="4"/>
              </svg>
            </div>
          )}
        </div>

        {/* Name + Email + Upload buttons */}
        <div style={{flex:1}}>
          <div style={{fontWeight:800,fontSize:22,marginBottom:4,letterSpacing:-0.5}}>{profile?.full_name||'User'}</div>
          <div style={{fontSize:13,color:T.dim,marginBottom:16,fontFamily:"'JetBrains Mono',monospace"}}>{session?.user?.email}</div>
          <div style={{display:'flex',gap:8}}>
            <button onClick={()=>!uploading&&fileRef.current?.click()} disabled={uploading} style={{padding:'9px 18px',borderRadius:10,fontSize:12,fontWeight:700,background:uploading?'rgba(255,255,255,0.04)':'#fff',color:uploading?T.dim:'#0b0b0b',border:'none',cursor:uploading?'wait':'pointer',fontFamily:T.font,transition:'all 0.3s',display:'inline-flex',alignItems:'center',gap:6}}>
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
              {uploading?'Uploading...':'Upload Photo'}
            </button>
            {profile?.avatar_url&&(
              <button onClick={handleRemove} style={{padding:'9px 16px',borderRadius:10,fontSize:12,fontWeight:600,background:'rgba(248,113,113,0.04)',color:T.danger,border:'1px solid rgba(248,113,113,0.08)',cursor:'pointer',fontFamily:T.font,transition:'all 0.3s',display:'inline-flex',alignItems:'center',gap:6}} onMouseEnter={e=>{e.currentTarget.style.background='rgba(248,113,113,0.1)'}} onMouseLeave={e=>{e.currentTarget.style.background='rgba(248,113,113,0.04)'}}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>
                Remove
              </button>
            )}
          </div>
          <input ref={fileRef} type="file" accept="image/png,image/jpeg,image/jpg,image/webp,image/gif" style={{display:'none'}} onChange={handleFile}/>
          <div style={{fontSize:10.5,color:T.dim,marginTop:10}}>PNG, JPG, WebP or GIF · Max 2MB</div>
        </div>
      </div>

      {/* Info Grid */}
      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:14}}>
        {[
          {label:'University',icon:'M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5',value:profile?.university||'—'},
          {label:'Branch',icon:'M4 19.5A2.5 2.5 0 016.5 17H20M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z',value:profile?.branch||'—'},
          {label:'Year of Study',icon:'M16 2v4M8 2v4M3 10h18M5 4h14a2 2 0 012 2v14a2 2 0 01-2 2H5a2 2 0 01-2-2V6a2 2 0 012-2z',value:profile?.year_of_study?`${profile.year_of_study}${['st','nd','rd','th'][profile.year_of_study-1]} Year`:'—'},
          {label:'Submissions',icon:'M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8zM14 2v6h6',value:'—'},
        ].map((item,i)=><div key={item.label} style={{padding:'16px 18px',borderRadius:12,background:'rgba(255,255,255,0.015)',border:`1px solid ${T.border}`,display:'flex',alignItems:'center',gap:14,transition:'all 0.3s',animation:`cardReveal 0.5s ease ${0.2+i*0.08}s both`}} onMouseEnter={e=>{e.currentTarget.style.borderColor='rgba(255,255,255,0.08)';e.currentTarget.style.background='rgba(255,255,255,0.03)'}} onMouseLeave={e=>{e.currentTarget.style.borderColor=T.border;e.currentTarget.style.background='rgba(255,255,255,0.015)'}}>
          <div style={{width:36,height:36,borderRadius:10,background:'rgba(255,255,255,0.03)',border:`1px solid ${T.border}`,display:'flex',alignItems:'center',justifyContent:'center',flexShrink:0}}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.5)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d={item.icon}/></svg>
          </div>
          <div style={{minWidth:0,flex:1}}>
            <div style={{fontSize:10,color:T.dim,textTransform:'uppercase',letterSpacing:1,fontWeight:600,marginBottom:2}}>{item.label}</div>
            <div style={{fontSize:14,fontWeight:700,color:'#fff',whiteSpace:'nowrap',overflow:'hidden',textOverflow:'ellipsis'}}>{item.value}</div>
          </div>
        </div>)}
      </div>
    </Card>
  </div>;
};

/* Reset Password Page — shown when user clicks reset link from email */
const ResetPasswordPage=({go,show})=>{
  const[pw,setPw]=useState('');
  const[cpw,setCpw]=useState('');
  const[ld,setLd]=useState(false);
  const[er,setEr]=useState({});

  const handleReset=async()=>{
    const x={};
    if(!valPw(pw).valid)x.pw="8+ chars: upper, lower, number, special";
    if(pw!==cpw)x.cpw="Passwords don't match";
    setEr(x);if(Object.keys(x).length)return;
    setLd(true);
    try{
      const{error}=await supabase.auth.updateUser({password:pw});
      if(error)throw error;
      show("Password updated successfully!","success");
      go("dashboard");
    }catch(err){
      setEr({pw:err.message||"Failed to update password"});
    }finally{setLd(false)}
  };

  return<AuthWrap title="Set New Password" sub="Enter your new password below">
    <Input label="New Password" type="password" value={pw} onChange={v=>setPw(v.target.value)} placeholder="Enter new password" error={er.pw} icon="🔒"/>
    <Input label="Confirm Password" type="password" value={cpw} onChange={v=>setCpw(v.target.value)} placeholder="Re-enter new password" error={er.cpw} icon="🔒"/>
    {pw&&<div style={{marginBottom:16,display:'flex',flexWrap:'wrap',gap:5}}>{Object.entries(valPw(pw)).filter(([k])=>k!=="valid").map(([k,v])=><span key={k} style={{fontSize:10.5,padding:'3px 10px',borderRadius:20,background:v?'rgba(74,222,128,0.06)':'rgba(248,113,113,0.06)',color:v?T.success:'rgba(248,113,113,0.6)'}}>{v?"✓":"✗"} {k}</span>)}</div>}
    <Btn onClick={handleReset} loading={ld} disabled={ld}>Update Password</Btn>
  </AuthWrap>;
};

/* Settings Tab — working password change */
const SetTab=({show})=>{
  const[newPw,setNewPw]=useState('');
  const[cfPw,setCfPw]=useState('');
  const[ld,setLd]=useState(false);
  const[er,setEr]=useState({});
  const[done,setDone]=useState(false);

  const handleChange=async()=>{
    const x={};
    if(!valPw(newPw).valid)x.newPw="8+ chars: upper, lower, number, special";
    if(newPw!==cfPw)x.cfPw="Passwords don't match";
    setEr(x);if(Object.keys(x).length)return;
    setLd(true);
    try{
      const{error}=await supabase.auth.updateUser({password:newPw});
      if(error)throw error;
      show("Password updated!","success");
      setNewPw('');setCfPw('');setDone(true);
      setTimeout(()=>setDone(false),3000);
    }catch(err){
      const msg=err.message||"";
      if(msg.includes("same")||msg.includes("different")){setEr({newPw:"New password must be different from the current one"})}
      else{setEr({newPw:msg||"Failed to update password"})}
    }finally{setLd(false)}
  };

  return<div><div style={{marginBottom:28}}><h1 style={{fontSize:26,fontWeight:800}}>Settings</h1></div><div style={{maxWidth:500,display:'flex',flexDirection:'column',gap:18}}>
    <Card animate style={{padding:28}}>
      <div style={{fontWeight:700,fontSize:15,marginBottom:16}}>Change Password</div>
      {done&&<div style={{background:'rgba(74,222,128,0.06)',border:'1px solid rgba(74,222,128,0.1)',borderRadius:10,padding:'11px 14px',marginBottom:16,color:T.success,fontSize:12.5}}>Password updated successfully!</div>}
      <Input label="New Password" type="password" value={newPw} onChange={v=>setNewPw(v.target.value)} placeholder="Enter new password" error={er.newPw} icon="🔒"/>
      <Input label="Confirm New Password" type="password" value={cfPw} onChange={v=>setCfPw(v.target.value)} placeholder="Re-enter new password" error={er.cfPw} icon="🔒"/>
      {newPw&&<div style={{marginBottom:16,display:'flex',flexWrap:'wrap',gap:5}}>{Object.entries(valPw(newPw)).filter(([k])=>k!=="valid").map(([k,v])=><span key={k} style={{fontSize:10.5,padding:'3px 10px',borderRadius:20,background:v?'rgba(74,222,128,0.06)':'rgba(248,113,113,0.06)',color:v?T.success:'rgba(248,113,113,0.6)'}}>{v?"✓":"✗"} {k}</span>)}</div>}
      <Btn onClick={handleChange} loading={ld} disabled={ld}>Update Password</Btn>
    </Card>
    <Card animate delay={0.1} style={{padding:28,border:'1px solid rgba(248,113,113,0.06)'}}>
      <div style={{fontWeight:700,color:T.danger,marginBottom:8}}>Danger Zone</div>
      <p style={{fontSize:13,color:T.muted,marginBottom:16}}>Delete account permanently.</p>
      <Btn v="danger" full={false} style={{padding:'9px 22px',fontSize:12.5}}>Delete Account</Btn>
    </Card>
  </div></div>;
};

/* Premium 3D Upload Option Card */
const UploadOptionCard=({type,title,desc,badge,onClick})=>{
  const[hov,setHov]=useState(false);
  const[tilt,setTilt]=useState({x:0,y:0});
  const[shine,setShine]=useState({x:50,y:50});
  const ref=useRef(null);

  const onMove=(e)=>{
    if(!ref.current)return;
    const r=ref.current.getBoundingClientRect();
    const x=((e.clientX-r.left)/r.width-0.5)*2;
    const y=((e.clientY-r.top)/r.height-0.5)*2;
    setTilt({x:x*10,y:-y*10});
    setShine({x:(x+1)*50,y:(y+1)*50});
  };

  const isRecord=type==='record';

  return <div ref={ref} onClick={onClick} onMouseEnter={()=>setHov(true)} onMouseLeave={()=>{setHov(false);setTilt({x:0,y:0})}} onMouseMove={onMove}
    style={{
      position:'relative',padding:'40px 28px 44px',borderRadius:20,overflow:'hidden',cursor:'pointer',
      background:'linear-gradient(155deg, rgba(255,255,255,0.02), rgba(255,255,255,0.005))',
      border:`1px solid ${hov?'rgba(255,255,255,0.15)':'rgba(255,255,255,0.05)'}`,
      transformStyle:'preserve-3d',
      transform:`rotateY(${tilt.x}deg) rotateX(${tilt.y}deg) ${hov?'translateY(-4px)':''}`,
      transition:tilt.x===0?'all 0.5s cubic-bezier(0.23,1,0.32,1)':'transform 0.12s ease-out, border-color 0.4s, box-shadow 0.4s',
      boxShadow:hov?'0 20px 60px rgba(0,0,0,0.4), 0 0 40px rgba(255,255,255,0.04)':'none',
      animation:`optIn 0.7s cubic-bezier(0.34,1.56,0.64,1) ${isRecord?0.1:0.25}s both`,
    }}>

    {/* Top accent line */}
    <div style={{position:'absolute',top:0,left:0,right:0,height:2,background:'linear-gradient(90deg,transparent,rgba(255,255,255,0.5),transparent)',transform:hov?'translateX(0)':'translateX(-100%)',transition:'transform 0.8s cubic-bezier(0.23,1,0.32,1)'}}/>

    {/* Scan line */}
    {hov&&<div style={{position:'absolute',left:0,right:0,height:1.5,background:'linear-gradient(90deg,transparent,rgba(255,255,255,0.4),transparent)',boxShadow:'0 0 8px rgba(255,255,255,0.3)',animation:'scanMove 2.5s linear infinite',pointerEvents:'none'}}/>}

    {/* Mouse shine */}
    <div style={{position:'absolute',inset:0,pointerEvents:'none',background:`radial-gradient(circle at ${shine.x}% ${shine.y}%, rgba(255,255,255,0.08) 0%, transparent 50%)`,opacity:hov?1:0,transition:'opacity 0.4s'}}/>

    {/* Circuit dots */}
    <div style={{position:'absolute',inset:0,pointerEvents:'none',opacity:hov?1:0,transition:'opacity 0.5s',backgroundImage:'radial-gradient(circle, rgba(255,255,255,0.08) 1px, transparent 1px)',backgroundSize:'20px 20px',maskImage:'radial-gradient(circle at 50% 80%, rgba(0,0,0,0.5), transparent 70%)',WebkitMaskImage:'radial-gradient(circle at 50% 80%, rgba(0,0,0,0.5), transparent 70%)'}}/>

    {/* Icon with rings */}
    <div style={{position:'relative',width:88,height:88,margin:'0 auto 24px',transform:hov?'translateZ(50px) scale(1.1)':'translateZ(30px)',transition:'transform 0.5s cubic-bezier(0.34,1.56,0.64,1)'}}>
      {/* Pulsing rings */}
      {[0,1,2].map(i=><div key={i} style={{position:'absolute',inset:-12-i*12,borderRadius:'50%',border:'1px solid rgba(255,255,255,0.08)',opacity:hov?1:0,transition:'opacity 0.4s',animation:hov?`ringPulse 2s ease-in-out ${i*0.4}s infinite`:'none'}}/>)}

      {/* Icon box */}
      <div style={{position:'relative',width:'100%',height:'100%',borderRadius:22,background:hov?'linear-gradient(135deg, rgba(255,255,255,0.15), rgba(255,255,255,0.03))':'linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02))',border:`1px solid ${hov?'rgba(255,255,255,0.25)':'rgba(255,255,255,0.1)'}`,display:'flex',alignItems:'center',justifyContent:'center',transition:'all 0.5s',overflow:'hidden',boxShadow:hov?'0 12px 32px rgba(255,255,255,0.08), inset 0 1px 0 rgba(255,255,255,0.15)':'none'}}>
        {/* Inner glow */}
        <div style={{position:'absolute',inset:0,background:'radial-gradient(circle at 50% 30%, rgba(255,255,255,0.15), transparent 60%)',opacity:hov?1:0,transition:'opacity 0.5s'}}/>

        {/* Icon SVG */}
        {isRecord?(
          <svg width="42" height="42" viewBox="0 0 48 48" fill="none" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" style={{position:'relative',zIndex:1,filter:hov?'drop-shadow(0 0 10px rgba(255,255,255,0.5))':'none',transition:'filter 0.5s'}}>
            <rect x="6" y="14" width="28" height="20" rx="3" style={{animation:hov?'camBreathe 2.5s ease-in-out infinite':'none',transformOrigin:'center'}}/>
            <path d="M34 20 L42 16 L42 32 L34 28 Z"/>
            <circle cx="14" cy="24" r="2.5" fill="#ff4444" stroke="none" style={{animation:hov?'recPulse 1s ease-in-out infinite':'recPulseSoft 2s ease-in-out infinite'}}/>
            <path d="M21 20 Q 24 24 21 28" opacity="0.5" style={{animation:hov?'waveExpand 1.5s ease-out infinite':'none'}}/>
            <path d="M25 18 Q 30 24 25 30" opacity="0.3" style={{animation:hov?'waveExpand 1.5s ease-out 0.3s infinite':'none'}}/>
          </svg>
        ):(
          <svg width="42" height="42" viewBox="0 0 48 48" fill="none" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" style={{position:'relative',zIndex:1,filter:hov?'drop-shadow(0 0 10px rgba(255,255,255,0.5))':'none',transition:'filter 0.5s'}}>
            <path d="M14 30 L14 40 A4 4 0 0 0 18 44 L30 44 A4 4 0 0 0 34 40 L34 30"/>
            <g style={{animation:hov?'arrowBounce 1.5s ease-in-out infinite':'none'}}>
              <path d="M24 30 L24 8"/>
              <polyline points="14 18 24 8 34 18"/>
            </g>
          </svg>
        )}
      </div>
    </div>

    {/* Title */}
    <div style={{fontSize:18,fontWeight:700,color:'#fff',textAlign:'center',marginBottom:6,letterSpacing:-0.3,transform:'translateZ(20px)',position:'relative'}}>{title}</div>

    {/* Desc */}
    <div style={{fontSize:11,color:hov?'rgba(255,255,255,0.6)':'rgba(255,255,255,0.35)',textAlign:'center',fontFamily:"'JetBrains Mono',monospace",letterSpacing:0.5,transition:'color 0.5s',transform:'translateZ(15px)',position:'relative'}}>{desc}</div>

    {/* Badge */}
    <div style={{position:'absolute',bottom:14,left:'50%',transform:`translateX(-50%) translateZ(10px)`,display:'flex',alignItems:'center',gap:6,padding:'4px 12px',borderRadius:100,background:'rgba(255,255,255,0.04)',border:'1px solid rgba(255,255,255,0.06)',fontSize:10,fontWeight:600,color:'rgba(255,255,255,0.4)',fontFamily:"'JetBrains Mono',monospace",letterSpacing:1,textTransform:'uppercase',opacity:hov?1:0,transition:'opacity 0.4s'}}>
      <span style={{width:5,height:5,borderRadius:'50%',background:'#4ade80',boxShadow:'0 0 6px rgba(74,222,128,0.6)',animation:'pulse 1.5s ease infinite'}}/>
      {badge}
    </div>

    {/* Corner accent */}
    <div style={{position:'absolute',bottom:0,right:0,width:100,height:100,pointerEvents:'none',background:'radial-gradient(circle at bottom right, rgba(255,255,255,0.05), transparent 70%)',opacity:hov?1:0,transition:'opacity 0.5s'}}/>
  </div>;
};

/* Upload Page */
const UploadPg=({go,show,session,profile,ls})=>{const[mode,setMode]=useState(null),[file,setFile]=useState(null),[drag,setDrag]=useState(false);const[upl,setUpl]=useState(false),[prog,setProg]=useState(0),[stage,setStage]=useState("");const[error,setError]=useState(null),[prev,setPrev]=useState(null),[isRec,setIsRec]=useState(false);const fr=useRef();const hf=f=>{const v=valFile(f);if(!v.ok){setError(v.err);show(v.err,"error");return}setError(null);setFile(f);setPrev(URL.createObjectURL(f))};
  const upload=async()=>{if(!file||!session)return;setUpl(true);setProg(0);setStage("Uploading video...");try{const cloud=await uploadToCloudinary(file,pct=>{setProg(Math.round(pct*0.4));setStage(`Uploading... ${pct}%`)});setStage("Saving to database...");setProg(42);const sub=await db.createSubmission(session.user.id,cloud.secure_url,file.name,file.size,isRec?"record":"upload");setProg(45);setStage("Starting AI analysis...");api.setToken(session.access_token);try{const job=await api.startProcessing({video_url:cloud.secure_url,submission_id:sub.id,user_id:session.user.id,user_name:profile?.full_name||"Student",user_university:profile?.university||"",user_branch:profile?.branch||"",user_year:profile?.year_of_study||1});setProg(50);await api.waitForCompletion(job.job_id,s=>{const backendProg=s.progress||0;setProg(50+Math.round(backendProg*0.5));setStage(s.message||"Processing...")},5000);show("Resume & clip ready!","success")}catch(e){console.error(e);show("Uploaded! Processing queued.","info")}await ls(session.user.id);go("dashboard")}catch(err){console.error(err);show(err.message||"Failed","error");try{await ls(session.user.id)}catch{}}finally{setUpl(false)}};
  return<div style={{minHeight:'100vh',display:'flex',alignItems:'center',justifyContent:'center',padding:24,position:'relative',zIndex:2}}><Card style={{width:'100%',maxWidth:560,padding:36,animation:'fadeUp .4s ease'}} glow><div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:28}}><div><h2 style={{fontSize:22,fontWeight:800}}>New Submission</h2><p style={{color:T.dim,fontSize:13,marginTop:4}}>Record or upload</p></div>{!upl&&<Btn v="ghost" onClick={()=>go("dashboard")} full={false}>← Back</Btn>}</div>{upl&&prog>=45&&<FaceScan stage={stage} prog={Math.min(Math.round((prog-45)*100/55),100)}/>}{!upl&&!mode&&!file&&<div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:16,marginBottom:20,perspective:1200}}>
  <UploadOptionCard type="record" title="Record" desc="WEBCAM + MIC" badge="Live" onClick={()=>setMode('record')}/>
  <UploadOptionCard type="upload" title="Upload" desc="MP4 · WEBM · MOV" badge="200MB Max" onClick={()=>setMode('upload')}/>
</div>}{!upl&&mode==="record"&&!file&&<Recorder onDone={f=>{setFile(f);setPrev(URL.createObjectURL(f));setIsRec(true);setMode("preview")}} onCancel={()=>setMode(null)}/>}{!upl&&mode==="upload"&&!file&&<><div onDragOver={e=>{e.preventDefault();setDrag(true)}} onDragLeave={()=>setDrag(false)} onDrop={e=>{e.preventDefault();setDrag(false);if(e.dataTransfer.files.length)hf(e.dataTransfer.files[0])}} onClick={()=>fr.current?.click()} style={{border:`2px dashed ${drag?'rgba(255,255,255,0.15)':T.border}`,borderRadius:14,padding:'48px 24px',textAlign:'center',cursor:'pointer',marginBottom:16}}><svg width="38" height="38" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{margin:'0 auto 12px',opacity:0.25,display:'block',color:'#fff'}}><polygon points="23 7 16 12 23 17"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg><p style={{color:T.textSoft,fontSize:14}}>Drag & drop or click</p><p style={{color:T.dim,fontSize:12}}>Max 200MB</p><input ref={fr} type="file" accept="video/mp4,video/webm,video/quicktime" style={{display:'none'}} onChange={e=>{if(e.target.files[0])hf(e.target.files[0])}}/></div>{error&&<div style={{background:'rgba(248,113,113,0.04)',border:'1px solid rgba(248,113,113,0.06)',borderRadius:10,padding:'11px 14px',marginBottom:16,color:T.danger,fontSize:12.5}}>{error}</div>}<Btn v="secondary" onClick={()=>setMode(null)}>← Back</Btn></>}{file&&<>{!upl&&<div style={{borderRadius:14,overflow:'hidden',marginBottom:16,background:'#000'}}><video src={prev} controls style={{width:'100%',maxHeight:280,display:'block'}}/></div>}<div style={{background:T.inputBg,borderRadius:12,padding:'14px 18px',marginBottom:16,display:'flex',justifyContent:'space-between',alignItems:'center',border:`1px solid ${T.border}`}}><div><div style={{fontSize:13.5,fontWeight:600}}>{file.name}</div><div style={{fontSize:11.5,color:T.dim,marginTop:2}}>{(file.size/(1024*1024)).toFixed(1)} MB</div></div>{!upl&&<span onClick={()=>{setFile(null);setPrev(null);setMode(null);setIsRec(false)}} style={{color:T.dim,cursor:'pointer',fontSize:20}}>×</span>}</div>{upl&&prog<45&&<div style={{marginBottom:16}}><div style={{display:'flex',justifyContent:'space-between',marginBottom:7}}><span style={{color:T.muted,fontSize:13}}>{stage}</span><span style={{color:'#fff',fontSize:13,fontWeight:700}}>{Math.round(prog)}%</span></div><div style={{height:5,background:'rgba(255,255,255,0.03)',borderRadius:10,overflow:'hidden'}}><div style={{height:'100%',borderRadius:10,background:T.gradient,width:`${prog}%`,transition:'width .5s'}}/></div></div>}<Btn onClick={upload} disabled={upl} loading={upl}>{upl?"Processing...":"Upload & Process →"}</Btn>{upl&&<button onClick={()=>{setUpl(false);setProg(0);setStage('');setFile(null);setPrev(null);setMode(null);setIsRec(false);go('dashboard');show('Upload cancelled','info')}} style={{width:'100%',marginTop:10,padding:'11px 0',borderRadius:12,fontSize:13,fontWeight:600,background:'transparent',color:T.danger,border:`1px solid rgba(248,113,113,0.12)`,cursor:'pointer',fontFamily:T.font,transition:'all 0.3s',display:'flex',alignItems:'center',justifyContent:'center',gap:6}} onMouseEnter={e=>{e.target.style.background='rgba(248,113,113,0.06)';e.target.style.borderColor='rgba(248,113,113,0.25)'}} onMouseLeave={e=>{e.target.style.background='transparent';e.target.style.borderColor='rgba(248,113,113,0.12)'}}>
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
  Cancel
</button>}</>}</Card></div>};
