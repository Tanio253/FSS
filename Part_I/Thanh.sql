CREATE TABLE Orders (
    id SERIAL PRIMARY KEY,
    foodID VARCHAR(4) CHECK (foodID BETWEEN 'F001' AND 'F008'),
    customerID VARCHAR(4) CHECK (customerID BETWEEN 'C001' AND 'C050'),
    employeeID VARCHAR(4) CHECK (employeeID BETWEEN 'E001' AND 'E098'),
    quantity INT CHECK (quantity > 0),
    timeStart TIMESTAMP,
    timeEnd TIMESTAMP,
    dishCondition VARCHAR(20) CHECK (dishCondition IN ('Rare', 'Medium Rare', 'Medium', 'Medium Well', 'Well Done')),
    status VARCHAR(20) CHECK (status IN ('Received', 'Cooking', 'Done', 'Cancelled')),
    tips DECIMAL(5,2) CHECK (tips >= 0)
);

INSERT INTO Orders (foodID, customerID, employeeID, quantity, timeStart, timeEnd, dishCondition, status, tips)
SELECT 
    'F' || LPAD((floor(random() * 8) + 1)::TEXT, 3, '0'), 
    'C' || LPAD((floor(random() * 50) + 1)::TEXT, 3, '0'), 
    'E' || LPAD((floor(random() * 98) + 1)::TEXT, 3, '0'), 
    (random() * 5 + 1)::INT, 
    NOW() - (random() * INTERVAL '30 days'),
    NOW() - (random() * INTERVAL '30 days'),
    (ARRAY['Rare', 'Medium Rare', 'Medium', 'Medium Well', 'Well Done'])[floor(random() * 5) + 1],
    (ARRAY['Received', 'Cooking', 'Done', 'Cancelled'])[floor(random() * 4) + 1],
    round((random() * 20)::numeric, 2) 
FROM generate_series(1, 1000);
