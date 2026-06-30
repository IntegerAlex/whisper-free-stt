/**
 * Spring animation utility — creates natural, bouncy transitions.
 * Based on the damped harmonic oscillator model used by Framer Motion.
 *
 * Usage:
 *   const spring = createSpring({ stiffness: 400, damping: 0.7 });
 *   spring.setTarget(1.0);
 *   // In requestAnimationFrame loop:
 *   const value = spring.step(dt);
 */

export interface SpringConfig {
  stiffness: number;  // Spring constant (higher = snappier)
  damping: number;    // Damping ratio (0 = no damping, 1 = critically damped)
  mass?: number;      // Mass (default 1)
  precision?: number; // Stop threshold (default 0.001)
}

interface SpringState {
  value: number;
  velocity: number;
  target: number;
}

export function createSpring(config: SpringConfig) {
  const { stiffness, damping, mass = 1, precision = 0.001 } = config;
  const state: SpringState = { value: 0, velocity: 0, target: 0 };
  let rafId: number | null = null;
  let lastTime: number | null = null;
  let onUpdate: ((value: number) => void) | null = null;
  let onSettle: (() => void) | null = null;

  function step(dt: number) {
    // Clamp dt to prevent huge jumps (e.g. when tab is backgrounded)
    const clampedDt = Math.min(dt, 0.064);

    const displacement = state.value - state.target;
    const springForce = -stiffness * displacement;
    const dampingForce = -damping * 2 * Math.sqrt(stiffness) * state.velocity;
    const acceleration = (springForce + dampingForce) / mass;

    state.velocity += acceleration * clampedDt;
    state.value += state.velocity * clampedDt;

    // Check if settled
    if (
      Math.abs(state.velocity) < precision &&
      Math.abs(state.value - state.target) < precision
    ) {
      state.value = state.target;
      state.velocity = 0;
      return true; // settled
    }
    return false;
  }

  function tick(time: number) {
    if (lastTime === null) {
      lastTime = time;
      rafId = requestAnimationFrame(tick);
      return;
    }
    const dt = (time - lastTime) / 1000;
    lastTime = time;

    const settled = step(dt);
    onUpdate?.(state.value);

    if (settled) {
      rafId = null;
      lastTime = null;
      onSettle?.();
      return;
    }
    rafId = requestAnimationFrame(tick);
  }

  return {
    /** Set the target value. Animation starts automatically. */
    setTarget(target: number) {
      state.target = target;
      if (rafId === null) {
        lastTime = null;
        rafId = requestAnimationFrame(tick);
      }
    },

    /** Set the current value directly (no animation). */
    setValue(value: number) {
      state.value = value;
      state.velocity = 0;
      onUpdate?.(value);
    },

    /** Get the current value. */
    getValue() {
      return state.value;
    },

    /** Register update callback. */
    onUpdate(cb: (value: number) => void) {
      onUpdate = cb;
    },

    /** Register settle callback (called when animation finishes). */
    onSettle(cb: () => void) {
      onSettle = cb;
    },

    /** Stop animation immediately. */
    stop() {
      if (rafId !== null) {
        cancelAnimationFrame(rafId);
        rafId = null;
        lastTime = null;
      }
    },

    /** Check if currently animating. */
    isAnimating() {
      return rafId !== null;
    },
  };
}

// Pre-configured spring presets
export const SPRING_SNAPPY = { stiffness: 400, damping: 0.7 };
export const SPRING_SOFT = { stiffness: 200, damping: 0.8 };
export const SPRING_BOUNCY = { stiffness: 300, damping: 0.5 };
