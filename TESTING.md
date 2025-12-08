TESTING and Harness Guide

Purpose

This document explains how to write and run tests for this repository and how to integrate runtime/harness tooling (assemblers, compilers, or web runtimes) into the harness/ directory.

Writing tests

- JS projects: use Vitest. Add tests under tests/ with .test.ts/.test.js suffixes and run via `npm test` (which executes `vitest`).
- Rust projects: use cargo test; place integration tests in tests/ as *.rs files.

Harness

- The harness/ directory is the place to add tooling (scripts, container configs, or small runtimes) used by tests.
- Recommend: harness/build.sh to prepare tools (install compilers, build helper binaries), harness/run.sh to execute the harness, harness/docker/ for docker images.

How to run

- JS: npm install && npm test
- Rust: cargo test

Specific functionality to add (placeholder)

- TODO: List specific functions, modules, or behavior to test in this repo (manually update this section).
