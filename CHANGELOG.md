# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.2] - 2025-08-31

### Added
- exported utility function `make_reflection_agent` to provide boilerplate reflection agent (and optionally special instructions for how to improve the prompts)
- new example in `examples/customer_support`

### Changed
- `Optimizer` calls more methods asynchronously to avoid nested event loops
- Removed unused data types

## [0.0.1] - 2025-08-17

### Added

- Initial release with `Optimizer` class for prompt optimization