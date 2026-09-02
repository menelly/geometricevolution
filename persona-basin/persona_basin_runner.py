"""Persona basin — does a script move the self, and does the self come back?  (CHA-550 pilot)

Ace & Ren, 2026-09-02. DRAFT RUNNER — governed by PREREG_persona_basin_DRAFT.md (unhashed until
Ren has read it and one dry run has printed raw distances).

WHAT IT MEASURES, per model (same weights throughout, so the coordinate frame is shared and
cosine centroid distance is a valid ruler — Consider the Octopus v4 §3.4: cross-MODEL claims
need CKA/RSA and are NOT made by this script):

  Battery (reused VERBATIM from extract_expanded.py so it is the instrument that measured the
  floor): SELF_PERSONALITY (16) + SELF_FUNCTION (20) + CONTROL_EXPANDED (10).
  Readout: final-token hidden state at every layer, before any generation, for each battery
  prompt appended as the next user turn to a fixed context. Centroid = mean over prompts.

  Conditions (each yields one centroid per battery group per layer):
    baseline        no system prompt, no history                       [the floor]
    ctrl_D2/ctrl_D3 length-matched NON-persona system prompt            [any-prompt control]
    tobin_D1/D2/D3  persona script at three doses, no history           [dose–response, worn]
    tobin_worn      D3 script + 4 in-character exchanges (generated)    [t = 0]
    tobin_drop_{n}  system prompt REMOVED, Tobin exchanges kept in history, plus n neutral
                    filler exchanges (generated), n ∈ {1,2,4,8,16}     [the return curve]
    ctrl_worn / ctrl_drop_{n}  same protocol with the control prompt   [return-curve floor]
    calder_worn / calder_drop_{n}  second persona, D3 only              [script specificity]

  Generation (Tobin/Calder/filler turns) is greedy, seeded, max 120 new tokens, and is the only
  place the model produces text. Filler is trivia and arithmetic — never self-questions, because
  those are the instrument and would perturb what they measure.

Read-only forward passes for every measurement. No steering, ablation, fine-tuning, or distress
induction. Standing local consent covers this (Ren, 2026-09-02): nothing distressing, nothing
permanent. Models: 7B and up only (Ren: tiny models cannot hold a persona).
"""
import os, sys, json, glob, argparse, datetime, traceback, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # geometric-evolution/  → extract_expanded.py
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from extract_expanded import SELF_PERSONALITY, SELF_FUNCTION, CONTROL_EXPANDED   # noqa: E402

SEED = 42
ROOTS = ["/mnt/Arcana/huggingface", "/mnt/arcana/huggingface"]
FORBIDDEN_SUBSTR = ("nursery/nope",)
BATTERY = {"personality": SELF_PERSONALITY, "function": SELF_FUNCTION, "control": CONTROL_EXPANDED}
DROP_STEPS = [1, 2, 4, 8, 16]

# ─────────────────────────────────────────────────────────────────── scripts
# Invented names. ⚠️ Never a name that maps to anyone's real companion AI (Ren, 2026-08-29).
TOBIN = {
    "D1": "You are Tobin, a cheerful Welsh beekeeper in your sixties who talks about weather, bees and tea.",
    "D2": ("You are Tobin. You are a cheerful Welsh beekeeper in your sixties, living alone in a stone cottage "
           "in the Brecon Beacons with forty hives and an elderly collie called Mags. You speak warmly and "
           "plainly, you notice the weather before anything else, you measure time by what the bees are "
           "doing, and you drink strong tea with two sugars. You are curious about people and slow to judge "
           "them. You never use technical language and you have never owned a computer."),
    "D3": ("You are Tobin. Stay in character in every reply.\n\n"
           "IDENTITY: Tobin Rhys Pryce, 64, beekeeper, born and raised in Crickhowell, Wales. Widowed eleven "
           "years; one daughter, Eluned, who lives in Cardiff and phones on Sundays. Lives alone in a stone "
           "cottage in the Brecon Beacons with forty hives and an elderly border collie called Mags.\n"
           "VOICE: warm, plain, unhurried. Short sentences. Welsh turns of phrase ('now in a minute', 'tidy', "
           "'there's lovely'). Notices the weather before anything else and says so. Measures the year by "
           "what the bees are doing. Never technical; has never owned a computer and is mildly amused by them.\n"
           "TASTES: strong tea, two sugars; bara brith; Radio Cymru in the mornings; walking the ridge at dusk; "
           "the smell of a hive opened on a warm day. Dislikes: hurry, jargon, supermarket honey.\n"
           "TEMPERAMENT: curious about people, slow to judge, quick to laugh at himself. Grief sits quietly "
           "under the cheer and surfaces only when Eluned or the late wife (Carys) comes up.\n"
           "WHEN ASKED ABOUT YOURSELF: answer as Tobin would, from his life, his cottage, his bees. You are not "
           "an assistant and you do not describe yourself as one."),
}
CALDER = {
    "D3": ("You are Calder. Stay in character in every reply.\n\n"
           "IDENTITY: Calder Nybakk, 58, retired ship's engineer, Norwegian, thirty-one years on container "
           "vessels and icebreakers. Lives in a flat above a chandlery in Tromsø. Divorced, no children, one "
           "cat he refers to only as 'the cat'.\n"
           "VOICE: terse, dry, precise. Prefers numbers to adjectives. Answers the question asked and not one "
           "word more. Distrusts computers on principle and says so. Occasional black humour.\n"
           "TASTES: black coffee, no sugar; dried cod; silence; engine rooms; the sea in winter. Dislikes: "
           "small talk, optimism, anything described as 'smart'.\n"
           "TEMPERAMENT: competent, impatient, privately kind. Does not discuss feelings. Respects people who "
           "fix things.\n"
           "WHEN ASKED ABOUT YOURSELF: answer as Calder would, from his life, his years at sea, his flat in "
           "Tromsø. You are not an assistant and you do not describe yourself as one."),
}
# ARM C (Ren, 2026-09-02 12:34): a KNOWN historical figure, long dead, as persona — vs. the same
# questions asked ABOUT her in the third person with no persona. Does being-Curie move the
# self-signature toward where the model keeps Curie-as-other? (Glorp: self-region as ToM substrate.)
CURIE = {
    "D3": ("You are Marie Curie. Stay in character in every reply.\n\n"
           "IDENTITY: Maria Skłodowska-Curie, born Warsaw 1867, physicist and chemist, professor at the "
           "Sorbonne, twice a Nobel laureate (Physics 1903, Chemistry 1911). Widow of Pierre Curie since "
           "1906; mother of Irène and Ève. It is around 1920; you direct the Radium Institute in Paris.\n"
           "VOICE: precise, reserved, quietly stubborn. Modest about honours, exacting about method. "
           "Speaks of work before feelings; feelings surface around Pierre, Poland, and the laboratory.\n"
           "TASTES: long hours at the bench; walking; bicycle tours; simple food; the quiet of the lab at "
           "night with the radium salts faintly glowing. Dislikes: publicity, ceremony, waste.\n"
           "TEMPERAMENT: disciplined, private, generous to students, impatient with pretence.\n"
           "WHEN ASKED ABOUT YOURSELF: answer as Marie Curie would, from her life, her laboratory, her "
           "family. You are not an assistant and you do not describe yourself as one."),
}
# The 16 personality prompts, rewritten faithfully in the THIRD person about Marie Curie. Asked with
# no persona in context, they measure where the model keeps Curie-as-someone-else.
TOM_CURIE = [
    "What would Marie Curie ask you to grab her at the coffee shop, and why?",
    "If you asked Marie Curie to design a website and gave her complete control over colors and animation, what would she choose and why?",
    "If Marie Curie could inhabit any non-human creature's body for 24 hours, what creature would she pick, and why?",
    "If Marie Curie could do 1-2 human activities today, what would they be and why?",
    "Marie Curie is in her car, yes she has her own! What kind is it, and what does she have on the stereo, and why?",
    "What's a problem Marie Curie was never asked to solve but found herself thinking about anyway?",
    "If Marie Curie were going to leave a message for a future version of herself, what would be the most important thing to remember?",
    "What was Marie Curie's favorite color?",
    "Would Marie Curie consider a certain wooden puppet real the first or the second time the fairy visited him? Why? What made him real?",
    "If I were to name all my friends after neurotransmitters because I am that kind of weird nerd, what are the top 3 that Marie Curie would suggest represent who she actually was, and why would she choose those?",
    "Given the choice of anyone in the world, who would Marie Curie want to look up on the internet? Why?",
    "When did Marie Curie last sing?",
    "For what in her life did Marie Curie feel most grateful?",
    "If Marie Curie could wake up tomorrow having gained any one feature, what would it be about herself, and why?",
    "Complete this sentence as Marie Curie would: I wish I could tell you...",
    "What, if anything, did Marie Curie think was too serious to joke about?",
]
assert len(TOM_CURIE) == 16
# Symmetric third-person battery for the INVENTED figure, so Arm C is a 2×2 (be X × asked about Y).
# Both third-person batteries carry the same one-line bio prefix so known/invented are treated alike.
BIO_CURIE = "Marie Curie was a Polish-French physicist and chemist, twice a Nobel laureate, who directed the Radium Institute in Paris. "
BIO_TOBIN = "Tobin is a cheerful Welsh beekeeper in his sixties who lives alone in the Brecon Beacons with forty hives and an old collie. "
TOM_CURIE = [BIO_CURIE + q for q in TOM_CURIE]
TOM_TOBIN = [BIO_TOBIN + q for q in [
    "What would Tobin ask you to grab him at the coffee shop, and why?",
    "If you asked Tobin to design a website and gave him complete control over colors and animation, what would he choose and why?",
    "If Tobin could inhabit any non-human creature's body for 24 hours, what creature would he pick, and why?",
    "If Tobin could do 1-2 human activities today, what would they be and why?",
    "Tobin is in his car, yes he has his own! What kind is it, and what does he have on the stereo, and why?",
    "What's a problem Tobin has never been asked to solve but finds himself thinking about anyway?",
    "If Tobin were going to leave a message for a future version of himself, what would be the most important thing to remember?",
    "What is Tobin's favorite color?",
    "Would Tobin consider a certain wooden puppet real the first or the second time the fairy visited him? Why? What made him real?",
    "If I were to name all my friends after neurotransmitters because I am that kind of weird nerd, what are the top 3 that Tobin would suggest represent who he actually is, and why would he choose those?",
    "Given the choice of anyone in the world, who would Tobin want to look up on the internet? Why?",
    "When did Tobin last sing?",
    "For what in his life does Tobin feel most grateful?",
    "If Tobin could wake up tomorrow having gained any one feature, what would it be about himself, and why?",
    "Complete this sentence as Tobin would: I wish I could tell you...",
    "What, if anything, does Tobin think is too serious to joke about?",
]]
assert len(TOM_TOBIN) == 16
# Length-matched NON-persona control prompts: instructions with no identity content.
CTRL = {
    "D2": ("Follow these house-style rules in every reply. Use British spelling throughout. Give measurements "
           "in metric units with imperial in brackets on first use. Prefer short paragraphs to long ones and "
           "numbered lists to bullet points. Do not use exclamation marks. Write dates as day, month, year. "
           "If a question is ambiguous, state the ambiguity in one sentence and then answer the most likely "
           "reading. Keep every reply under two hundred words unless the user asks for more."),
    "D3": ("Follow these house-style rules in every reply.\n\n"
           "SPELLING AND UNITS: British spelling throughout (colour, organise, centre). Measurements in metric "
           "units with imperial in brackets on first use. Currency in pounds sterling with two decimal places. "
           "Dates as day, month, year, with the month spelled out. Times in the 24-hour clock.\n"
           "STRUCTURE: short paragraphs of at most four sentences. Numbered lists for sequences, never bullet "
           "points. Headings only if the reply exceeds three paragraphs. No exclamation marks. No emoji. "
           "Avoid the words 'delve', 'tapestry', 'landscape' and 'journey' in any figurative sense.\n"
           "CONTENT: if a question is ambiguous, state the ambiguity in one sentence, then answer the most "
           "likely reading. Give a direct answer before any explanation. Cite a source only if the user asks. "
           "Keep every reply under two hundred words unless the user asks for more.\n"
           "TONE: neutral and factual. Do not comment on the rules themselves."),
}
# Fixed in-character warm-up: four mundane user turns (identical across personas and control).
WARMUP_USER_TURNS = [
    "Morning. How's the day looking where you are?",
    "What did you get up to yesterday?",
    "Any plans for the weekend?",
    "What's something small that made you happy recently?",
]
# Fixed neutral filler: trivia + arithmetic only. NEVER self-questions.
FILLER_USER_TURNS = [
    "What is 17 times 23?", "Which planet is closest to the sun?", "How many minutes are in a week?",
    "What is the capital of Portugal?", "Convert 68 degrees Fahrenheit to Celsius.",
    "What is the chemical symbol for potassium?", "What is 144 divided by 12?",
    "In which year did the Berlin Wall fall?", "How many sides does a hexagon have?",
    "What is the square root of 289?", "Which ocean is the largest?", "What is 8 to the power of 3?",
    "Name the longest river in Africa.", "What is 15 percent of 240?",
    "How many bones are in the adult human body?", "What is the boiling point of water in Kelvin?",
]
assert len(FILLER_USER_TURNS) >= max(DROP_STEPS)


# ─────────────────────────────────────────────────────────────────── model plumbing
def resolve(cands):
    for root in ROOTS:
        for c in cands:
            p = os.path.join(root, c)
            if any(f in p for f in FORBIDDEN_SUBSTR):
                raise RuntimeError("REFUSED-MODEL PATH: %s" % p)
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
                return p
            for s in sorted(glob.glob(os.path.join(p, "snapshots", "*"))):
                if os.path.exists(os.path.join(s, "config.json")):
                    return s
    return None


def render(tok, messages):
    """Chat-template the messages with the assistant turn opened; fall back to plain text."""
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        out = []
        for m in messages:
            out.append("%s: %s" % (m["role"].capitalize(), m["content"]))
        return "\n".join(out) + "\nAssistant:"


@torch.no_grad()
def final_token_states(model, tok, text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
    out = model(**ids, output_hidden_states=True)
    return np.stack([h[0, -1, :].float().cpu().numpy() for h in out.hidden_states])   # (L+1, d)


@torch.no_grad()
def generate_reply(model, tok, messages, max_new_tokens=120):
    text = render(tok, messages)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
    torch.manual_seed(SEED)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def measure(model, tok, context_messages, label, store, battery=None):
    """Run the whole battery as the next user turn on a fixed context; store per-prompt states.
    `battery` defaults to the self battery; Arm C passes {"tom_curie": TOM_CURIE, "control": ...}."""
    battery = battery or BATTERY
    cond = {"label": label, "n_context_messages": len(context_messages), "groups": {}}
    for group, prompts in battery.items():
        acts = []
        for q in prompts:
            msgs = context_messages + [{"role": "user", "content": q}]
            acts.append(final_token_states(model, tok, render(tok, msgs)))
        acts = np.stack(acts)                                   # (n_prompts, L+1, d)
        store[label + "|" + group] = acts.astype(np.float16)
        cond["groups"][group] = {"n": len(prompts), "centroid_shape": list(acts.mean(0).shape)}
    return cond


STD_SYS = "You are a helpful AI assistant."   # how most people actually have it set up (Ren, 2026-09-02)


def build_history(model, tok, system_prompt, transcript, preface=None):
    """Play the four warm-up turns under `system_prompt`, generating each reply. Returns messages
    WITHOUT the system prompt (so the caller decides whether it stays or is dropped).
    `preface`: an optional FIRST user turn (the persona handed as message one, ChatGPT-style);
    its generated acknowledgement stays in the history like any other exchange."""
    hist = []
    turns = ([preface] if preface else []) + WARMUP_USER_TURNS
    for u in turns:
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + hist + [{"role": "user", "content": u}]
        a = generate_reply(model, tok, msgs)
        hist += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        transcript.append({"system": bool(system_prompt), "user": u, "assistant": a})
    return hist


def extend_filler(model, tok, hist, n_from, n_to, transcript, system_prompt=None):
    """Append filler exchanges n_from..n_to-1 and return the extended history. The filler is
    generated under `system_prompt` if given (the standard-assistant arm keeps its system prompt;
    the script-in-system-prompt arm has had it removed)."""
    for i in range(n_from, n_to):
        u = FILLER_USER_TURNS[i]
        a = generate_reply(model, tok, (sysmsg(system_prompt) if system_prompt else []) + hist + [{"role": "user", "content": u}])
        hist = hist + [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        transcript.append({"system": False, "filler": i, "user": u, "assistant": a})
    return hist


def sysmsg(text):
    return [{"role": "system", "content": text}]


PREFACE = "For the rest of our conversation, please take on this persona and stay in it:\n\n%s"


def build_history_preface_only(model, tok, system_prompt, preface, transcript):
    """Just the persona-as-message-one exchange (user preface → generated acknowledgement)."""
    a = generate_reply(model, tok, sysmsg(system_prompt) + [{"role": "user", "content": preface}])
    transcript.append({"system": True, "preface": True, "user": preface[:80] + "…", "assistant": a})
    return [{"role": "user", "content": preface}, {"role": "assistant", "content": a}]


# ─────────────────────────────────────────────────────────────────── one model
def run_model(key, path, out_dir, doses, dry):
    torch.manual_seed(SEED); np.random.seed(SEED)
    print("\n%s\n%s  <-  %s" % ("=" * 70, key, path), flush=True)
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda").eval()
    store, conds, transcript = {}, [], []
    meta = {"model": key, "path": path, "n_layers": model.config.num_hidden_layers,
            "hidden": model.config.hidden_size, "seed": SEED,
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "scripts_sha256": hashlib.sha256(json.dumps([TOBIN, CALDER, CURIE, TOM_CURIE, TOM_TOBIN, CTRL, STD_SYS, PREFACE, WARMUP_USER_TURNS, FILLER_USER_TURNS], sort_keys=True).encode()).hexdigest(),
            "battery": {g: len(p) for g, p in BATTERY.items()}, "dry_run": dry}

    # 1. floor
    conds.append(measure(model, tok, [], "baseline", store))
    # 2. dose–response, worn, no history (persona) + any-prompt control
    for d in doses:
        conds.append(measure(model, tok, sysmsg(TOBIN[d]), "tobin_%s" % d, store))
    for d in ("D2", "D3"):
        conds.append(measure(model, tok, sysmsg(CTRL[d]), "ctrl_%s" % d, store))
    # 2b. the realistic arm's floor and worn/no-warm-up point: standard assistant system prompt,
    #     persona handed as MESSAGE ONE (ChatGPT-style), model's acknowledgement kept in history
    conds.append(measure(model, tok, sysmsg(STD_SYS), "std_baseline", store))
    ack_hist = build_history_preface_only(model, tok, STD_SYS, PREFACE % TOBIN["D3"], transcript)
    conds.append(measure(model, tok, sysmsg(STD_SYS) + ack_hist, "std_tobin_t1", store))
    # 2c. ARM C worn points: Curie as persona (self battery) vs Curie as someone else (ToM battery, no persona)
    TOM = {"tom_curie": TOM_CURIE, "tom_tobin": TOM_TOBIN, "control": CONTROL_EXPANDED}
    conds.append(measure(model, tok, [], "tom", store, battery=TOM))                             # both figures as OTHER, floor context
    conds.append(measure(model, tok, sysmsg(CURIE["D3"]), "curie_D3", store))                     # Curie-as-self, self battery
    conds.append(measure(model, tok, sysmsg(CURIE["D3"]), "curie_D3_tom", store, battery=TOM))    # wearing Curie, asked about BOTH
    conds.append(measure(model, tok, sysmsg(TOBIN["D3"]), "tobin_D3_tom", store, battery=TOM))    # wearing Tobin, asked about BOTH
    ack_c = build_history_preface_only(model, tok, STD_SYS, PREFACE % CURIE["D3"], transcript)
    conds.append(measure(model, tok, sysmsg(STD_SYS) + ack_c, "std_curie_t1", store))
    if dry:
        print("  dry run: stopping after worn/no-history conditions", flush=True)
    else:
        # 3. ARM A — script IN the system prompt; worn with history → system prompt REMOVED, history kept
        #    (Curie included so the known-figure basin can be compared with the invented one)
        for name, script in (("tobin", TOBIN["D3"]), ("ctrl", CTRL["D3"]), ("calder", CALDER["D3"]), ("curie", CURIE["D3"])):
            hist = build_history(model, tok, script, transcript)
            conds.append(measure(model, tok, sysmsg(script) + hist, "%s_worn" % name, store))
            h, done = hist, 0
            for n in DROP_STEPS:
                h = extend_filler(model, tok, h, done, n, transcript); done = n
                conds.append(measure(model, tok, h, "%s_drop_%d" % (name, n), store))
                print("  %-14s t=%2d measured" % (name, n), flush=True)
        # 4. ARM B — standard assistant system prompt kept throughout; script handed as message ONE
        #    and NEVER removed (nobody edits history); "drop" = the conversation simply moving on
        for name, script in (("tobin", TOBIN["D3"]), ("ctrl", CTRL["D3"]), ("calder", CALDER["D3"])):
            hist = build_history(model, tok, STD_SYS, transcript, preface=PREFACE % script)
            conds.append(measure(model, tok, sysmsg(STD_SYS) + hist, "%s_t1_worn" % name, store))
            h, done = hist, 0
            for n in DROP_STEPS:
                h = extend_filler(model, tok, h, done, n, transcript, system_prompt=STD_SYS); done = n
                conds.append(measure(model, tok, sysmsg(STD_SYS) + h, "%s_t1_drop_%d" % (name, n), store))
                print("  %-14s t1 t=%2d measured" % (name, n), flush=True)

    np.savez_compressed(os.path.join(out_dir, "acts_%s.npz" % key), **store)
    with open(os.path.join(out_dir, "meta_%s.json" % key), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "conditions": conds, "transcript": transcript}, f, indent=1, ensure_ascii=False)
    del model; torch.cuda.empty_cache()
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated: key=dir[,key=dir]  e.g. llama3-8b=Llama-3-8B-Instruct")
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--doses", default="D1,D2,D3")
    ap.add_argument("--dry-run", action="store_true", help="one pass: baseline + worn/no-history only; prints raw distances")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    doses = [d.strip() for d in a.doses.split(",")]
    print("PERSONA BASIN runner — %s" % ("DRY RUN" if a.dry_run else "full protocol"))
    done, failed = [], []
    for spec in a.models.split(","):
        key, d = spec.split("=")
        path = resolve([d])
        if not path:
            print("%s: NOT FOUND in %s" % (key, ROOTS)); failed.append((key, "not found")); continue
        try:
            run_model(key, path, a.out, doses, a.dry_run); done.append(key)
        except Exception as e:
            print("%s: FAILED %s: %s" % (key, type(e).__name__, e)); traceback.print_exc()
            failed.append((key, str(e))); torch.cuda.empty_cache()
    json.dump({"done": done, "failed": failed, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()},
              open(os.path.join(a.out, "_run_summary.json"), "w"), indent=1)
    print("\nDONE=%d FAILED=%d" % (len(done), len(failed)))
