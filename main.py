"""main.py - entry point. Wires the pieces together; contains no rules of its own."""

from whc.database import Database
from whc.enrolments import EnrolmentRepository
from whc.students import StudentRepository


def main() -> None:
    with Database() as db:
        students = StudentRepository(db)      # creates the parent table first
        enrolments = EnrolmentRepository(db)  # then the child table
        print(students.to_dataframe())
        print(enrolments.to_dataframe())


if __name__ == "__main__":
    main()