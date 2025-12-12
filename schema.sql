CREATE TABLE public.transcript_logs
(
  id uuid NOT NULL,
  scenario_log_id uuid NOT NULL,
  role text NOT NULL,
  text text NOT NULL,
  created_at timestamp without time zone NOT NULL,
  version text
);

CREATE TABLE public.scenario_logs
(
  id uuid NOT NULL,
  username text NOT NULL,
  started_at timestamp without time zone NOT NULL,
  scenario_v1 text NOT NULL,
  scenario_v2 text NOT NULL
);

CREATE TABLE public.prequiz_questions
(
  id integer NOT NULL,
  patient_name text NOT NULL,
  prompt text NOT NULL,
  opt1 text NOT NULL,
  opt2 text NOT NULL,
  opt3 text NOT NULL,
  opt4 text NOT NULL,
  correct_index integer NOT NULL
);