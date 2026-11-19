"""main.py - entry point. Wires the pieces together; contains no rules of its own."""

from whc.database import Database
from whc.enrolments import EnrolmentRepository
from whc.fitness_tests import FitnessTestRepository
from whc.students import StudentRepository
from whc.test_results import TestResultRepository


def main() -> None:
    with Database() as db:
        students = StudentRepository(db)          # creates the parent table first
        enrolments = EnrolmentRepository(db)      # then the child table
        fitness_tests = FitnessTestRepository(db) # the fitness test catalogue
        fitness_tests.load_catalogue()            # populates it
        test_results = TestResultRepository(db)   # the test results table
        print(students.to_dataframe())
        print(enrolments.to_dataframe())
        print(fitness_tests.to_dataframe())
        print(test_results.to_dataframe())


if __name__ == "__main__":
    main()