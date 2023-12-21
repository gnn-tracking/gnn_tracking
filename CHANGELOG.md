# Changelog

This changelog mostly collects important changes to the models that are not
backward compatible but result in different results.

## 23.12.1

### Breaking changes

* #475 (changed normalization of repulsive hinge loss for GC)

## 23.12.0

### Breaking changes

* #465 (changed behavior of condensation losses)
* #466 (change normalization for residual connections)

### API changes

* #467 (changes configuration of residual connections)

## 23.10.0


Important fixes:

* #437 (changing metric definition!)
* #432 (fixes geometric graph building!)
* #431 (fixes fast potential loss)
* #439, #444

and more

## 23.09.0

* Important fixes to residual networks (applying ReLUs correctly)
* Option to run loss function for OC in memory-efficient way
