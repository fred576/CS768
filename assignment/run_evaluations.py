import subprocess

def run_single_eval(a1, a2):
    completed_process = subprocess.run(
        ["python", "evaluation.py", "--test-paper-title", str(a1), "--test-paper-abstract", str(a2)],
        capture_output=True,  # Capture stdout
        text=True             # Get output as string, not bytes
    )

    # Get the result (it will be in stdout)
    result = completed_process.stdout.strip()
    return result

if __name__ == "__main__":
    t1 = """Computationally Efficient RL under Linear Bellman Completeness for Deterministic Dynamics"""
    a1 = """We study computationally and statistically efficient Reinforcement Learning algorithms for the linear Bellman Complete setting. This setting uses linear function approximation to capture value functions and unifies existing models like linear Markov Decision Processes (MDP) and Linear Quadratic Regulators (LQR). While it is known from the prior works that this setting is statistically tractable, it remained open whether a computationally efficient algorithm exists. Our work provides a computationally efficient algorithm for the linear Bellman complete setting that works for MDPs with large action spaces, random initial states, and random rewards but relies on the underlying dynamics to be deterministic. Our approach is based on randomization: we inject random noise into least squares regression problems to perform optimistic value iteration. Our key technical contribution is to carefully design the noise to only act in the null space of the training data to ensure optimism while circumventing a subtle error amplification issue."""

    output = run_single_eval(t1, a1)
    print(f"Result from evaluation.py: \n{output}")
    # assign_marks_to_preds(output)
