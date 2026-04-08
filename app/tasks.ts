/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Task } from "./models.ts";

export const TASKS: Task[] = [
  {
    id: "easy",
    name: "Stable System",
    description: "Few machines, stable system, no failures. Focus on basic scheduling.",
    config: {
      num_machines: 2,
      initial_jobs: 10,
      energy_budget: 50.0,
      failure_rate: 0.0,
      max_steps: 50,
    },
  },
  {
    id: "medium",
    name: "Energy Constraints",
    description: "More machines and jobs with tight energy constraints. Balance throughput and power.",
    config: {
      num_machines: 4,
      initial_jobs: 20,
      energy_budget: 80.0,
      failure_rate: 0.05,
      max_steps: 100,
    },
  },
  {
    id: "hard",
    name: "Dynamic Factory",
    description: "High machine failure rate, dynamic job arrivals, and priority jobs. Manage health and delays.",
    config: {
      num_machines: 8,
      initial_jobs: 30,
      energy_budget: 150.0,
      failure_rate: 0.15,
      max_steps: 200,
      intermittent_failure_rate: 0.05,
      degradation_rate: 0.01,
      high_intensity_multiplier: 3.0,
    },
  },
];
