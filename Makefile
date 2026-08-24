# Delegates to ./newton, which runs everything in the container.
#
#   make help          what there is to run
#   make test
#   make arch DOC=pedagogy
#   make sweep KNOB=prerequisites
#
# Anything without a target here is still reachable: ./newton <verb> [flags].

NEWTON ?= ./newton
DOC    ?= architecture
KNOB   ?= arbitration
ARM    ?= coupled

.PHONY: help build rebuild test typecheck check version arch docs config results \
        smoke verifier cohort paired propagation ordering coverage power calibrate \
        sweep figures all demo diagnostic tutor sitting shell up-models down

help:        ; @$(NEWTON) help
build:       ; @$(NEWTON) build
rebuild:     ; @$(NEWTON) rebuild

test:        ; @$(NEWTON) test
typecheck:   ; @$(NEWTON) typecheck
check:       ; @$(NEWTON) check
version:     ; @$(NEWTON) version

arch:        ; @$(NEWTON) arch $(DOC)
docs:        ; @$(NEWTON) docs
config:      ; @$(NEWTON) config
results:     ; @$(NEWTON) results

smoke:       ; @$(NEWTON) smoke
verifier:    ; @$(NEWTON) verifier
cohort:      ; @$(NEWTON) cohort $(ARM)
paired:      ; @$(NEWTON) paired
propagation: ; @$(NEWTON) propagation
ordering:    ; @$(NEWTON) ordering
coverage:    ; @$(NEWTON) coverage
power:       ; @$(NEWTON) power
calibrate:   ; @$(NEWTON) calibrate
sweep:       ; @$(NEWTON) sweep $(KNOB)
figures:     ; @$(NEWTON) figures
all:         ; @$(NEWTON) all

demo:        ; @$(NEWTON) demo
diagnostic:  ; @$(NEWTON) diagnostic
tutor:       ; @$(NEWTON) tutor

sitting:     ; @$(NEWTON) sitting
shell:       ; @$(NEWTON) shell

up-models:   ; @$(NEWTON) up-models
down:        ; @$(NEWTON) down
