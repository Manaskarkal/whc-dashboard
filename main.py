"""main.py - entry point. Wires all tables and prints a clean summary."""

from whc.audit import AuditRepository
from whc.auth import AuthRepository
from whc.database import Database
from whc.enrolments import EnrolmentRepository
from whc.fitness_tests import FitnessTestRepository
from whc.students import StudentRepository
from whc.test_results import TestResultRepository


def main() -> None:
    with Database() as db:
        students = StudentRepository(db)
        enrolments = EnrolmentRepository(db)
        fitness_tests = FitnessTestRepository(db)
        fitness_tests.load_catalogue()
        test_results = TestResultRepository(db)
        audit = AuditRepository(db)
        auth = AuthRepository(db, audit=audit)
        print(f"Students:      {len(students.to_dataframe())} rows")
        print(f"Enrolments:    {len(enrolments.to_dataframe())} rows")
        print(f"Fitness Tests: {len(fitness_tests.to_dataframe())} rows")
        print(f"Test Results:  {len(test_results.to_dataframe())} rows")
        print(f"Users:         {len(auth.to_dataframe())} rows")
        print(f"Audit Log:     {len(audit.to_dataframe())} rows (chain: {audit.verify_chain()[1]})")


if __name__ == "__main__":
    main()
