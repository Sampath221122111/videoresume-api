import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL;
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY;

// sessionStorage = session dies when tab/browser closes = auto-logout
export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
  auth: {
    storage: window.sessionStorage,
    autoRefreshToken: true,
    persistSession: true,
    detectSessionInUrl: true,
  },
});

export const auth = {
  async signUp(email, password) {
    const { data, error } = await supabase.auth.signUp({ email, password });
    if (error) throw error;
    return data;
  },
  async signIn(email, password) {
    const { data, error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) throw error;
    return data;
  },
  async signOut() {
    const { error } = await supabase.auth.signOut();
    if (error) throw error;
  },
  async resetPassword(email) {
    const { data, error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: `${window.location.origin}`,
    });
    if (error) throw error;
    return data;
  },
  async getSession() {
    const { data: { session } } = await supabase.auth.getSession();
    return session;
  },
  onAuthStateChange(callback) {
    return supabase.auth.onAuthStateChange(callback);
  },
};

export const db = {
  async getProfile(userId) {
    const { data, error } = await supabase.from('profiles').select('*').eq('id', userId).single();
    if (error && error.code !== 'PGRST116') throw error;
    return data;
  },
  async updateProfile(userId, updates) {
    const { data, error } = await supabase.from('profiles').update(updates).eq('id', userId).select().single();
    if (error) throw error;
    return data;
  },
  async createSubmission(userId, videoUrl, filename, fileSize, method) {
    const { data, error } = await supabase.from('submissions').insert({
      user_id: userId, video_url: videoUrl, video_filename: filename,
      video_size_bytes: fileSize, upload_method: method, status: 'pending',
    }).select().single();
    if (error) throw error;
    return data;
  },
  async getSubmissions(userId) {
    const { data, error } = await supabase.from('submissions')
      .select('*, extracted_skills(skill_name)').eq('user_id', userId)
      .order('created_at', { ascending: false });
    if (error) throw error;
    return data || [];
  },
  async deleteSubmission(id) {
    const { error } = await supabase.from('submissions').delete().eq('id', id);
    if (error) throw error;
  },
};
