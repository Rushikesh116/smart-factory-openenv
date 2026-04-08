/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import express from "express";
import cors from "cors";
import path from "path";
import { createServer as createViteServer } from "vite";
import { SmartFactoryEnv } from "./app/env.ts";
import { Action, ActionType } from "./app/models.ts";
import { TASKS } from "./app/tasks.ts";
import { SmartFactoryGrader } from "./app/grader.ts";
import { BaselineAgent } from "./app/baseline_agent.ts";

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(cors());
  app.use(express.json());

  const envs: Record<string, SmartFactoryEnv> = {};
  let currentTask = TASKS[0];

  app.get("/health", (_req, res) => {
    res.json({ ok: true, task: currentTask.id, initialized: Object.keys(envs).length > 0 });
  });

  // OpenEnv Endpoints
  app.post("/reset", (req, res) => {
    const taskId = req.body.task_id || "easy";
    const sessionId = req.body.session_id || req.query.session_id || "default";
    currentTask = TASKS.find((t) => t.id === taskId) || TASKS[0];
    envs[sessionId] = new SmartFactoryEnv(currentTask.config);
    const state = envs[sessionId].reset();
    res.json(state);
  });

  app.post("/step", (req, res) => {
    const sessionId = req.body.session_id || req.query.session_id || "default";
    const env = envs[sessionId];
    if (!env) {
      return res.status(400).json({ error: "Environment not initialized. Call /reset first." });
    }
    const action: Action = req.body.action;
    const result = env.step(action);
    res.json(result);
  });

  app.get("/state", (req, res) => {
    const sessionId = req.query.session_id || "default";
    const env = envs[sessionId as string];
    if (!env) {
      return res.status(400).json({ error: "Environment not initialized. Call /reset first." });
    }
    res.json(env.state());
  });

  app.get("/tasks", (req, res) => {
    res.json(TASKS);
  });

  app.get("/grader", (req, res) => {
    const sessionId = req.query.session_id || "default";
    const env = envs[sessionId as string];
    if (!env) {
      return res.status(400).json({ error: "Environment not initialized. Call /reset first." });
    }
    const score = SmartFactoryGrader.grade(env.state());
    res.json({ score });
  });

  app.get("/baseline", (req, res) => {
    const sessionId = req.query.session_id || "default";
    const env = envs[sessionId as string];
    if (!env) {
      return res.status(400).json({ error: "Environment not initialized. Call /reset first." });
    }
    const action = BaselineAgent.getAction(env.state());
    res.json({ action });
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
