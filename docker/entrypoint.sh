#!/usr/bin/env bash
#
# Verb dispatcher for the container. Each verb runs one command with the
# arguments the stored results were produced under; anything typed after the
# verb is appended, so a flag given here overrides the default.
#
#   newton help
#   newton paired --n 40      # same run, fewer learners
#
# Unrecognised verbs fall through to `agent-newton`, so the whole CLI stays
# reachable: `newton evaluate verifier --domain toy_algebra`.

set -euo pipefail

CALCULUS="experiments/configs/calculus.yaml"
DEMO="experiments/configs/demo.yaml"
SMOKE="experiments/configs/smoke.yaml"

# The paired comparison and the sweeps are reported at these values. Overridable
# for a quicker look, at the cost of not reproducing the stored numbers.
N="${NEWTON_N:-160}"
SEED="${NEWTON_SEED:-20260811}"

# The doubt sweep is the one experiment recorded under a different seed —
# results/sweep_doubt/summary.json says 20260819. Running it at the shared seed
# produces a coherent sweep that is not the stored one.
DOUBT_SEED="${NEWTON_SEED:-20260819}"

# Supplies the measured misclassification rate to the propagation study's
# `noised` condition. Committed, so it is present in a fresh clone.
DIAGNOSTIC_SUMMARY="results/diagnostic_calculus_gemma4-12b_think-false_labels-concept/summary.json"

MODEL="${NEWTON_MODEL:-gemma4:12b}"

# Where a re-run's summary goes. Not over the stored one: a committed summary
# names the run directories behind it, and rewriting it to name directories that
# were never committed leaves a quotable number whose origin is not inspectable.
# See results/README.md. Overriding is still one flag — `paired --out results/…`.
RERUNS="${NEWTON_OUT:-results/reruns}"

if [ -t 1 ]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; CYAN=$'\033[36m'; YELLOW=$'\033[33m'
  RED=$'\033[31m'; OFF=$'\033[0m'
else
  BOLD=""; DIM=""; CYAN=""; YELLOW=""; RED=""; OFF=""
fi

say()  { printf '%s\n' "$*"; }
step() { printf '\n%s── %s %s\n' "$BOLD" "$*" "$OFF"; }

# --- the model check ------------------------------------------------------
# Runs before anything model-backed. A container that cannot reach Ollama
# otherwise spends 120 s in the provider's timeout, three times over, before
# saying so.
require_model() {
  local host="${OLLAMA_HOST:-http://localhost:11434}"
  local tags
  if ! tags=$(curl -fsS --max-time 4 "${host}/api/tags" 2>/dev/null); then
    cat >&2 <<MSG
${RED}no Ollama at ${host}${OFF}

  This command needs a model. Three ways to give it one:

    1. Start Ollama on the host and pull the model:
         ollama serve
         ollama pull ${MODEL}
       The container reaches it at host.docker.internal by default.

    2. Run one in a container instead:
         ./newton up-models

    3. Point at a server elsewhere:
         OLLAMA_HOST=http://<host>:11434 ./newton ${1:-demo}

  Everything else here is model-free — see ${CYAN}newton help${OFF}.
MSG
    exit 1
  fi
  if ! printf '%s' "$tags" | grep -q "\"${MODEL%%:*}"; then
    say "${YELLOW}${MODEL} is not pulled on ${host}${OFF}"
    say "  ollama pull ${MODEL}${DIM}   (or, for the container: ./newton pull)${OFF}"
    exit 1
  fi
}

# Run an experiment into RERUNS, then compare against the stored summary if
# there is one. The question an examiner has is not what the numbers are, it is
# whether they are the numbers in the write-up.
# reproduce <stored-dir-name> <paths-not-produced> <command...>
reproduce() {
  local name="$1"; shift
  local ignore="$1"; shift
  local out="$RERUNS/$name"
  "$@" --out "$out"

  # Every JSON the run wrote that the stored directory also has. The sweeps
  # write one per parameter, and coverage and power do not call theirs
  # summary.json, so matching by name covers all of them.
  local stored="results/$name" produced base
  [ -d "$stored" ] || return 0
  for produced in "$out"/*.json; do
    [ -e "$produced" ] || continue
    base=$(basename "$produced")
    [ -f "$stored/$base" ] || continue
    python experiments/compare_summary.py \
      --rerun "$produced" --stored "$stored/$base" --ignore "$ignore" || true
  done
}

usage() {
  cat <<MSG
${BOLD}Agent_Newton${OFF} — multi-agent tutoring system with a shared learner-state layer

  ${DIM}newton <verb> [flags]      flags after a verb are passed straight through${OFF}

${BOLD}Read the design${OFF}
  ${CYAN}arch${OFF}              the component reference, rendered
  ${CYAN}arch <name>${OFF}       any other file under docs/ — pedagogy, learner_state, …
  ${CYAN}docs${OFF}              list what is there
  ${CYAN}config${OFF}            show the run configurations and validate them
  ${CYAN}results${OFF}           the stored summaries, and what each one holds

${BOLD}Check it${OFF}
  ${CYAN}test${OFF}              the test suite, minus the model-backed cases
  ${CYAN}typecheck${OFF}         pyright
  ${CYAN}check${OFF}             domain content validation, then every run config
  ${CYAN}version${OFF}           versions and the commit this image was built from

${BOLD}Run an experiment${OFF}  ${DIM}— no model involved; summaries land in ${RERUNS}/${OFF}
  ${CYAN}smoke${OFF}             end-to-end on toy_algebra, seconds
  ${CYAN}verifier${OFF}          symbolic verifier against its gold set, under a second
  ${CYAN}cohort [arm]${OFF}      one cohort — coupled (default) or decoupled
  ${CYAN}paired${OFF}            the paired comparison, n=${N}, dose-matched
  ${CYAN}propagation${OFF}       diagnostic error propagated to outcomes, all conditions
  ${CYAN}ordering${OFF}          the ordering probe
  ${CYAN}coverage${OFF}          misconception coverage against the item budget
  ${CYAN}power${OFF}             power analysis ${DIM}(the long one — minutes)${OFF}
  ${CYAN}calibrate${OFF}         mastery estimate against held-out performance
  ${CYAN}sweep <knob>${OFF}      arbitration | prerequisites | headroom | doubt
  ${CYAN}figures${OFF}           redraw the figures from the stored summaries
  ${CYAN}all${OFF}               every model-free experiment above, in order

${BOLD}Needs a model${OFF}  ${DIM}— host Ollama at \$OLLAMA_HOST, or ./newton up-models${OFF}
  ${CYAN}demo${OFF}              work through a session yourself, blackboard visible
  ${CYAN}diagnostic${OFF}        score the diagnostic agent against injected labels
  ${CYAN}tutor${OFF}             score the tutor on the turns a learner would read

${BOLD}Inspect a session${OFF}
  ${CYAN}sitting [run]${OFF}     read a stored sitting back as prose ${DIM}(default: latest)${OFF}
  ${CYAN}history <learner>${OFF} what was taught, per concept, across sittings ${DIM}(after a demo)${OFF}
  ${CYAN}shell${OFF}             a shell in the container

${BOLD}From the host${OFF}  ${DIM}— ./newton only; these have no meaning inside${OFF}
  ${CYAN}build${OFF} / ${CYAN}rebuild${OFF}   build the image; rebuild ignores the cache
  ${CYAN}dev <verb>${OFF}        run against the working tree, without rebuilding
  ${CYAN}up-models${OFF}         start a containerised Ollama and pull the model
  ${CYAN}pull [model]${OFF}      pull another model into it
  ${CYAN}down${OFF}              stop it

  ${DIM}An experiment writes to ${RERUNS}/ and is compared against the stored${OFF}
  ${DIM}summary as it finishes. The committed ones are never overwritten;${OFF}
  ${DIM}--out results/<name> does that deliberately.${OFF}

  ${DIM}OLLAMA_HOST=${OLLAMA_HOST:-unset}   NEWTON_N=${N}   NEWTON_SEED=${SEED}${OFF}
MSG
}

verb="${1:-help}"
[ $# -gt 0 ] && shift || true

case "$verb" in
  help|--help|-h)   usage ;;

  # --- read ---------------------------------------------------------------
  arch)             exec python /usr/local/lib/newton/render_doc.py "${1:-architecture}" ;;
  docs)             exec python /usr/local/lib/newton/render_doc.py ;;
  config)
    for cfg in experiments/configs/*.yaml; do
      step "$cfg"
      agent-newton config-check "$cfg"
    done
    ;;
  results)
    # -not -path excludes the per-run directories, which are named by timestamp:
    # those are provenance, not a reported number.
    summaries=$(find results -maxdepth 2 \
      \( -name 'summary*.json' -o -name 'coverage.json' -o -name 'power.json' \) \
      -not -path 'results/20*' | sort)
    step "aggregate summaries"
    printf '%s\n' "$summaries"
    step "run directories those summaries name"
    # The chain results/README.md asks to be kept whole: a summary naming a run
    # that is not present has a quotable number with no inspectable origin.
    printf '%s\n' "$summaries" \
      | xargs grep -ho '20[0-9]\{6\}T[0-9]\{6\}_[a-z0-9_.-]*' 2>/dev/null \
      | sort -u
    ;;

  # --- check --------------------------------------------------------------
  # cache_dir under /tmp: /app is not writable by the run-time user, and the
  # cache is worth nothing across container invocations anyway.
  test)             exec pytest -m "not slow" -q -o cache_dir=/tmp/pytest_cache "$@" ;;
  typecheck)        exec pyright "$@" ;;
  check)
    step "domain content"
    agent-newton domain validate all
    step "run configurations"
    for cfg in experiments/configs/*.yaml; do agent-newton config-check "$cfg"; done
    ;;
  version)
    agent-newton version
    say "python     $(python --version 2>&1 | cut -d' ' -f2)"
    say "git sha    ${AGENT_NEWTON_GIT_SHA:-<not recorded>}${AGENT_NEWTON_GIT_DIRTY:+ (working tree dirty)}"
    ;;

  # --- model-free runs ----------------------------------------------------
  smoke)            exec python experiments/run_cohort.py --config "$SMOKE" --arm coupled "$@" ;;
  verifier)
    reproduce "verifier_calculus" "" agent-newton evaluate verifier --domain calculus "$@"
    ;;
  cohort)
    arm="coupled"
    case "${1:-}" in coupled|decoupled) arm="$1"; shift ;; esac
    exec python experiments/run_cohort.py --config "$CALCULUS" --arm "$arm" "$@"
    ;;
  paired)
    reproduce "paired_calculus" "" python experiments/run_paired.py \
      --config "$CALCULUS" --n "$N" --seed "$SEED" --dose-matched "$@"
    ;;
  propagation)
    # The model-backed condition is left out of the default set, so the stored
    # summary has a branch this run does not produce.
    reproduce "propagation_calculus" "conditions.llm" python experiments/run_propagation.py \
      --config "$CALCULUS" \
      --diagnostic-summary "$DIAGNOSTIC_SUMMARY" \
      --conditions "oracle,noised,noised@0.10,noised@0.25,noised@0.50" "$@"
    ;;
  ordering)
    reproduce "ordering_calculus" "" python experiments/falsify_ordering.py \
      --config "$CALCULUS" --n "$N" --seed "$SEED" "$@"
    ;;
  coverage)
    reproduce "coverage_calculus" "" python experiments/measure_coverage.py \
      --config "$CALCULUS" "$@"
    ;;
  power)
    reproduce "power_calculus" "" python experiments/power_analysis.py \
      --config "$CALCULUS" "$@"
    ;;
  calibrate)
    reproduce "calibration_mastery" "" python experiments/calibrate_mastery.py \
      --config "$CALCULUS" --n "$N" --seed "$SEED" "$@"
    ;;
  sweep)
    knob="${1:-}"
    [ $# -gt 0 ] && shift || true
    case "$knob" in
      arbitration|prerequisites|headroom|doubt) : ;;
      *)
        say "${RED}sweep needs a knob${OFF}: arbitration | prerequisites | headroom | doubt"
        exit 1
        ;;
    esac
    if [ "$knob" = "doubt" ]; then knob_seed="$DOUBT_SEED"; else knob_seed="$SEED"; fi
    reproduce "sweep_${knob}" "" python "experiments/sweep_${knob}.py" \
      --config "$CALCULUS" --n "$N" --seed "$knob_seed" "$@"
    ;;
  figures)
    # The script's own default writes outside the repository; results/figures is
    # the tracked location, and the only one that exists here.
    exec python experiments/analysis/figures.py --out results/figures --format png "$@"
    ;;
  all)
    started=$SECONDS
    step "domain content and run configurations"
    agent-newton domain validate all
    for cfg in experiments/configs/*.yaml; do agent-newton config-check "$cfg"; done
    step "verifier gold set"
    reproduce "verifier_calculus" "" agent-newton evaluate verifier --domain calculus
    step "smoke cohort"
    python experiments/run_cohort.py --config "$SMOKE" --arm coupled
    step "misconception coverage"
    reproduce "coverage_calculus" "" python experiments/measure_coverage.py --config "$CALCULUS"
    step "paired comparison, n=$N"
    reproduce "paired_calculus" "" python experiments/run_paired.py \
      --config "$CALCULUS" --n "$N" --seed "$SEED" --dose-matched
    step "ordering probe"
    reproduce "ordering_calculus" "" python experiments/falsify_ordering.py \
      --config "$CALCULUS" --n "$N" --seed "$SEED"
    step "diagnostic error propagation"
    reproduce "propagation_calculus" "conditions.llm" python experiments/run_propagation.py \
      --config "$CALCULUS" --diagnostic-summary "$DIAGNOSTIC_SUMMARY" \
      --conditions "oracle,noised,noised@0.10,noised@0.25,noised@0.50"
    for knob in arbitration prerequisites headroom doubt; do
      step "sweep: $knob"
      if [ "$knob" = "doubt" ]; then knob_seed="$DOUBT_SEED"; else knob_seed="$SEED"; fi
      reproduce "sweep_${knob}" "" python "experiments/sweep_${knob}.py" \
        --config "$CALCULUS" --n "$N" --seed "$knob_seed"
    done
    step "mastery calibration"
    reproduce "calibration_mastery" "" python experiments/calibrate_mastery.py \
      --config "$CALCULUS" --n "$N" --seed "$SEED"
    step "power analysis"
    reproduce "power_calculus" "" python experiments/power_analysis.py --config "$CALCULUS"
    step "figures"
    python experiments/analysis/figures.py --out results/figures --format png
    step "done in $(( (SECONDS - started) / 60 ))m $(( (SECONDS - started) % 60 ))s"
    say "re-run summaries under ${CYAN}$RERUNS/${OFF}; the stored ones are untouched"
    say "${DIM}each is compared against results/ as it finishes — see the lines above${OFF}"
    ;;

  # --- model-backed -------------------------------------------------------
  demo)
    require_model demo
    exec agent-newton demo --config "$DEMO" "$@"
    ;;
  diagnostic)
    require_model diagnostic
    # The stored directory's name carries the model and the flags it was run
    # under, so it is named here rather than derived.
    reproduce "diagnostic_calculus_gemma4-12b_think-false_labels-concept" "" \
      agent-newton evaluate diagnostic --domain calculus --no-think "$@"
    ;;
  tutor)
    require_model tutor
    reproduce "tutor_calculus_gemma4-12b_think-false" "" \
      agent-newton evaluate tutor --domain calculus --no-think "$@"
    ;;

  # --- inspect ------------------------------------------------------------
  sitting)          exec agent-newton sitting "${1:-latest}" "${@:2}" ;;
  history)          exec agent-newton history "$@" ;;
  # --rcfile, because bash here starts with no PS1 at all: the prompt is
  # invisible and an interactive shell is indistinguishable from a hung one.
  shell|bash)       exec bash --rcfile /etc/newton.bashrc "$@" ;;

  # --- anything else is a CLI subcommand ----------------------------------
  *)                exec agent-newton "$verb" "$@" ;;
esac
