"""Run a Bell circuit through the existing FlagQuantum extension interface."""

import flagquantum as fq
from flagquantum.ecosystem.extensions.admission import (
    admit_backend_extension,
    withdraw_backend,
)

from flag_gems_quantum.integrations.flagquantum import FlagGemsQuantumBackend


def main() -> None:
    handle, capabilities = admit_backend_extension(FlagGemsQuantumBackend())
    try:
        # This first slice supports only zero-initialized IR programs and state
        # output. Circuit(inputs=...), measurements and gradients are deferred.
        program = fq.Circuit(2).h(0).cx(0, 1).to_ir()
        result = fq.run(
            program,
            options=fq.ExecutionOptions(
                backend=capabilities.name,
                device="cuda:0",
                mode="statevector",
                precision="complex64",
                require_gradients=False,
                allow_backend_fallback=False,
            ),
        )
        print(result.state)
        print("backend:", result.runtime["backend"])
    finally:
        handle.close()
        withdraw_backend(capabilities.name)


if __name__ == "__main__":
    main()
