-- ============================================================
-- AI Video Resume Generator — Supabase Database Migration
-- Run this in the Supabase SQL Editor (Dashboard → SQL Editor)
-- ============================================================

-- ── 1. Profiles table (extends auth.users) ──────────────────
CREATE TABLE IF NOT EXISTS profiles (
    id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
    full_name TEXT NOT NULL DEFAULT '',
    university TEXT DEFAULT '',
    branch TEXT DEFAULT '',
    year_of_study INT DEFAULT 1 CHECK (year_of_study BETWEEN 1 AND 6),
    profile_photo_url TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Auto-create profile when user signs up
CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO profiles (id, full_name)
    VALUES (NEW.id, COALESCE(NEW.raw_user_meta_data->>'full_name', ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION handle_new_user();


-- ── 2. Submissions table ────────────────────────────────────
CREATE TABLE IF NOT EXISTS submissions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES profiles(id) ON DELETE CASCADE NOT NULL,
    video_url TEXT NOT NULL,
    video_filename TEXT DEFAULT '',
    video_size_bytes BIGINT DEFAULT 0,
    upload_method TEXT DEFAULT 'upload' CHECK (upload_method IN ('upload', 'record')),
    transcript TEXT DEFAULT '',
    confidence_score FLOAT DEFAULT 0,
    energy_score FLOAT DEFAULT 0,
    expression_score FLOAT DEFAULT 0,
    eye_contact_score FLOAT DEFAULT 0,
    resume_pdf_url TEXT DEFAULT '',
    highlight_clip_url TEXT DEFAULT '',
    highlight_start FLOAT DEFAULT 0,
    highlight_end FLOAT DEFAULT 0,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    error_message TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast user-specific queries
CREATE INDEX IF NOT EXISTS idx_submissions_user_id ON submissions(user_id);
CREATE INDEX IF NOT EXISTS idx_submissions_status ON submissions(status);


-- ── 3. Extracted skills table ───────────────────────────────
CREATE TABLE IF NOT EXISTS extracted_skills (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    submission_id UUID REFERENCES submissions(id) ON DELETE CASCADE NOT NULL,
    skill_name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_skills_submission ON extracted_skills(submission_id);
CREATE INDEX IF NOT EXISTS idx_skills_name ON extracted_skills(skill_name);

-- Prevent duplicate skills per submission
CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_skill_per_submission
    ON extracted_skills(submission_id, skill_name);


-- ── 4. Row Level Security (RLS) ─────────────────────────────
-- Users can ONLY see and modify their OWN data

ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE submissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE extracted_skills ENABLE ROW LEVEL SECURITY;

-- Profiles: user can read/update only their own profile
DROP POLICY IF EXISTS "Users read own profile" ON profiles;
CREATE POLICY "Users read own profile" ON profiles
    FOR SELECT USING (auth.uid() = id);

DROP POLICY IF EXISTS "Users update own profile" ON profiles;
CREATE POLICY "Users update own profile" ON profiles
    FOR UPDATE USING (auth.uid() = id);

-- Submissions: user can CRUD only their own submissions
DROP POLICY IF EXISTS "Users read own submissions" ON submissions;
CREATE POLICY "Users read own submissions" ON submissions
    FOR SELECT USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users create own submissions" ON submissions;
CREATE POLICY "Users create own submissions" ON submissions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users delete own submissions" ON submissions;
CREATE POLICY "Users delete own submissions" ON submissions
    FOR DELETE USING (auth.uid() = user_id);

-- Backend (service role) can update any submission
-- Service role key bypasses RLS by default, so no policy needed for backend updates

-- Skills: user can read only skills from their own submissions
DROP POLICY IF EXISTS "Users read own skills" ON extracted_skills;
CREATE POLICY "Users read own skills" ON extracted_skills
    FOR SELECT USING (
        submission_id IN (
            SELECT id FROM submissions WHERE user_id = auth.uid()
        )
    );


-- ── 5. Updated_at auto-trigger ──────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS profiles_updated_at ON profiles;
CREATE TRIGGER profiles_updated_at
    BEFORE UPDATE ON profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS submissions_updated_at ON submissions;
CREATE TRIGGER submissions_updated_at
    BEFORE UPDATE ON submissions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();


-- ── 6. Helper view: submission with skills ──────────────────
CREATE OR REPLACE VIEW submission_details AS
SELECT
    s.*,
    p.full_name,
    p.university,
    p.branch,
    COALESCE(
        (SELECT json_agg(es.skill_name) FROM extracted_skills es WHERE es.submission_id = s.id),
        '[]'::json
    ) AS skills
FROM submissions s
JOIN profiles p ON p.id = s.user_id;


-- ============================================================
-- Done! Your database is ready.
-- ============================================================
