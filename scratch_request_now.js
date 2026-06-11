const http = require('http');

const data = JSON.stringify({
  text: "mua trà sữa hết 45k",
  profile: {
    budget_total: 5000000,
    budget_remain: 1200000,
    wallet_health: "can_than"
  },
  run_llm: true,
  nlg_persona: "dan_doi"
});

const options = {
  hostname: '127.0.0.1',
  port: 8000,
  path: '/api/v1/health',
  method: 'GET'
};

console.log("Sending GET request to http://127.0.0.1:8000/api/v1/health...");

const req = http.request(options, (res) => {
  let body = '';
  res.setEncoding('utf8');
  res.on('data', (chunk) => body += chunk);
  res.on('end', () => {
    console.log(`STATUS: ${res.statusCode}`);
    console.log("RESPONSE BODY:", body);
  });
});

req.on('error', (e) => {
  console.error(`Problem with request: ${e.message}`);
});

req.end();
