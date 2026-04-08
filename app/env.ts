/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { SmartFactorySimulator } from "./simulator.ts";
import { State, Action, StepResult } from "./models.ts";

export class SmartFactoryEnv {
  private simulator: SmartFactorySimulator;
  private config: any;

  constructor(config: any) {
    this.config = config;
    this.simulator = new SmartFactorySimulator(config);
  }

  public reset(): State {
    return this.simulator.reset();
  }

  public step(action: Action): StepResult {
    return this.simulator.step(action);
  }

  public state(): State {
    return this.simulator.getState();
  }
}
