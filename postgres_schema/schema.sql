-- PostgreSQL schema for Course-Job-Fit

CREATE TABLE Users_Students (
    user_id SERIAL PRIMARY KEY,
    first_name VARCHAR,
    last_name VARCHAR,
    email VARCHAR,
    major_id VARCHAR,
    gap DECIMAL,
    career_interest VARCHAR,
    password_hash VARCHAR,
    status VARCHAR
);

CREATE TABLE Programs (
    program_id SERIAL PRIMARY KEY,
    program_name VARCHAR,
    department VARCHAR,
    degree_level VARCHAR,
    major_minor_requirements TEXT,
    grad_program_prerequisites TEXT,
    description TEXT,
    program_url VARCHAR
);

CREATE TABLE User_Program (
    user_id INT REFERENCES Users_Students(user_id),
    program_id INT REFERENCES Programs(program_id),
    major_minor VARCHAR,
    start_date DATE,
    end_date DATE,
    catalog_year VARCHAR,
    status VARCHAR,
    PRIMARY KEY (user_id, program_id)
);

CREATE TABLE Courses (
    course_id SERIAL PRIMARY KEY,
    course_title VARCHAR,
    syllabus_text TEXT,
    prerequisites TEXT,
    credits INT,
    semester_offered VARCHAR,
    level VARCHAR,
    course_description TEXT,
    faculty_id INT,
    capacity INT,
    last_updated TIMESTAMP
);

CREATE TABLE User_Course (
    user_id INT REFERENCES Users_Students(user_id),
    course_id INT REFERENCES Courses(course_id),
    semester_taken VARCHAR,
    grade VARCHAR,
    status VARCHAR,
    PRIMARY KEY (user_id, course_id)
);

CREATE TABLE Skills (
    skill_id SERIAL PRIMARY KEY,
    skill_name VARCHAR,
    skill_description TEXT,
    skill_category VARCHAR
);

CREATE TABLE User_Skill (
    user_id INT REFERENCES Users_Students(user_id),
    skill_id INT REFERENCES Skills(skill_id),
    proficiency_level VARCHAR,
    last_assessed_date DATE,
    PRIMARY KEY (user_id, skill_id)
);

CREATE TABLE Faculty (
    skill_id INT PRIMARY KEY,
    skill_name VARCHAR,
    skill_description TEXT,
    skill_category VARCHAR
);

CREATE TABLE Course_Skill (
    course_id INT REFERENCES Courses(course_id),
    skill_id INT REFERENCES Skills(skill_id),
    coverage_level VARCHAR,
    PRIMARY KEY (course_id, skill_id)
);

CREATE TABLE Additional_Resources (
    resource_id SERIAL PRIMARY KEY,
    resource_name VARCHAR,
    resource_type VARCHAR,
    resource_description TEXT,
    contact_info VARCHAR,
    location TEXT,
    prerequisites TEXT,
    resource_url VARCHAR,
    membership_requirements TEXT
);

CREATE TABLE Program_Resource (
    program_id INT REFERENCES Programs(program_id),
    resource_id INT REFERENCES Additional_Resources(resource_id),
    is_mandatory BOOLEAN,
    notes TEXT,
    PRIMARY KEY (program_id, resource_id)
);

CREATE TABLE User_Resource (
    user_id INT REFERENCES Users_Students(user_id),
    resource_id INT REFERENCES Additional_Resources(resource_id),
    membership_date DATE,
    role VARCHAR,
    PRIMARY KEY (user_id, resource_id)
);

CREATE TABLE Job_Postings (
    job_id SERIAL PRIMARY KEY,
    employer VARCHAR,
    job_title VARCHAR,
    location VARCHAR,
    job_description TEXT,
    job_url VARCHAR,
    posting_date TIMESTAMP,
    expiration_date TIMESTAMP,
    job_type VARCHAR,
    min_salary DECIMAL,
    max_salary DECIMAL,
    application_deadline DATE
);

CREATE TABLE Job_Skill (
    job_id INT REFERENCES Job_Postings(job_id),
    skill_id INT REFERENCES Skills(skill_id),
    importance_level VARCHAR,
    PRIMARY KEY (job_id, skill_id)
);

CREATE TABLE Course_Job (
    course_id INT REFERENCES Courses(course_id),
    job_id INT REFERENCES Job_Postings(job_id),
    similarity_score DECIMAL,
    ranking INT,
    alignment_date TIMESTAMP,
    PRIMARY KEY (course_id, job_id)
);

CREATE TABLE Conversation_History (
    log_id SERIAL PRIMARY KEY,
    user_id INT REFERENCES Users_Students(user_id),
    timestamp TIMESTAMP,
    query_text TEXT,
    response_text TEXT,
    feedback_rating VARCHAR
);
