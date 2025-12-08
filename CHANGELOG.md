# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2025-12-05

### Added
- Production-ready initial release
- Core sparselist implementation with full list interface
- Sparse-optimized operations (sort, reverse, comparisons)
- Sparselist-specific methods: `compact()`, `unset()`
- Complete type hints support (PEP 561)
- Properties: `size`, `default`
- Memory-efficient storage using internal dictionary
- Support for Python 3.10+

## [1.1.0] - 2025-12-07

### Added
- Pickle support
- Initialize with all default arguments

## [1.2.0] - 2025-12-08

### Added
- Added __getstate__ for __setstate__ symmetry

## [1.3.0] - 2025-12-08

## Added
- Added reverse iterator support
