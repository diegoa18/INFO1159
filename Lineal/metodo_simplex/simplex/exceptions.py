from __future__ import annotations

from pathlib import Path

# exception -> LPerror -> los demas


class LPError(Exception):
    pass


class InvalidLPError(LPError):
    pass


class UnboundedError(LPError):
    pass


class InfeasibleError(LPError):  # infactible
    pass


class StabilityError(LPError):  # estabilidad, como: algo > MAX_ITERATIONS, eso
    pass


class TableauError(LPError):
    pass


class PivotError(LPError):  # para errores en el pivoteo
    pass


class InputParserError(LPError):
    pass


class InputValueError(InputParserError):
    @classmethod
    def positive_int(cls) -> InputValueError:
        return cls("Ingrese un entero positivo.")

    @classmethod
    def invalid_number(cls, text: str) -> InputValueError:
        return cls(f"'{text}' no es un numero valido")

    @classmethod
    def non_real(cls, text: str) -> InputValueError:
        return cls(f"'{text}' no es un numero real")

    @classmethod
    def non_finite(cls, text: str) -> InputValueError:
        return cls(f"'{text}' no es un numero finito")


class InputFormatError(InputParserError):
    @classmethod
    def objective_coeff_count(
        cls, expected: int, received: int
    ) -> InputFormatError:
        return cls(
            f"Se esperaban {expected} coeficientes para la funcion objetivo y llegaron {received}."
        )

    @classmethod
    def multiple_relations(cls) -> InputFormatError:
        return cls("La restriccion tiene mas de un signo de relacion.")

    @classmethod
    def missing_relation(cls) -> InputFormatError:
        return cls("La restriccion debe incluir <=, >= o =")

    @classmethod
    def invalid_constraint_format(
        cls, expected_coeffs: int, relation_symbol: str
    ) -> InputFormatError:
        return cls(
            f"Formato invalido. Use: a1 a2 ... a{expected_coeffs} {relation_symbol} b"
        )
