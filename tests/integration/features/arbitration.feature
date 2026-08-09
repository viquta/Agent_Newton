Feature: When evidence may revise the plan

  The policy decides when the planner is reopened. Triggers propose a replan;
  guardrails suppress one. A guardrail can never cause a replan, only prevent
  it, so no scenario below shows one doing otherwise.

  Background:
    Given a learner working on "distribute"
    And the replanning threshold is 0.15
    And a replan requires 2 items since the last one
    And a misconception must recur 2 times

  Scenario: no plan yet
    Given no concept has been planned
    When the policy is consulted
    Then it replans
    And the trigger is "no_plan"

  Scenario: the concept leaves the zone
    Given "distribute" is no longer in the frontier
    When the policy is consulted
    Then it replans
    And the trigger is "frontier_crossed"

  Scenario: leaving the zone cannot be suppressed
    Given "distribute" is no longer in the frontier
    And only 0 items have been worked since the last replan
    When the policy is consulted
    Then it replans
    And the trigger is "frontier_crossed"

  Scenario: mastery moves past the threshold
    Given 3 items have been worked since the last replan
    And mastery of "distribute" has moved by 0.25
    When the policy is consulted
    Then it replans
    And the trigger is "mastery_delta"

  Scenario: a movement below the threshold is not enough
    Given 3 items have been worked since the last replan
    And mastery of "distribute" has moved by 0.05
    When the policy is consulted
    Then it does not replan
    And no trigger fired

  Scenario: the rate limit holds back a real trigger
    Given 1 items have been worked since the last replan
    And mastery of "distribute" has moved by 0.25
    When the policy is consulted
    Then it does not replan
    And the trigger is "mastery_delta"
    And it was suppressed by "rate_limited"

  Scenario: a misconception that keeps recurring
    Given 3 items have been worked since the last replan
    And "distribute_first_term_only" has been confirmed 2 times
    When the policy is consulted
    Then it replans
    And the trigger is "misconception_repeat"

  Scenario: one occurrence is not a pattern
    Given 3 items have been worked since the last replan
    And "distribute_first_term_only" has been confirmed 1 times
    When the policy is consulted
    Then it does not replan

  Scenario: a diagnosis the verifier never confirmed does not count
    Given 3 items have been worked since the last replan
    And "distribute_first_term_only" has been diagnosed but not confirmed 3 times
    When the policy is consulted
    Then it does not replan

  Scenario: every replan carries the evidence that caused it
    Given 3 items have been worked since the last replan
    And mastery of "distribute" has moved by 0.25
    When the policy is consulted
    Then it replans
    And the evidence names the concept and the threshold
